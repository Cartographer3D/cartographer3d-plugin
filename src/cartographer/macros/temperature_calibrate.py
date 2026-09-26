from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, final

from typing_extensions import override

from cartographer.coil.calibration import fit_coil_temperature_model
from cartographer.interfaces.errors import McuDisconnectedError, PrinterShutdownError
from cartographer.interfaces.printer import GCodeDispatch, Macro, MacroParams, Mcu, ProbeMode, Sample, Toolhead
from cartographer.lib import scipy_helpers
from cartographer.lib.csv import generate_filepath, write_samples_to_csv
from cartographer.lib.log import log_duration
from cartographer.macros.fields import param, parse

if TYPE_CHECKING:
    from cartographer.interfaces.configuration import Configuration
    from cartographer.interfaces.multiprocessing import Scheduler, TaskExecutor

logger = logging.getLogger(__name__)

# Temperature monitoring constants
TEMP_CHECK_INTERVAL = 1.0  # Check temperature every second
PROGRESS_LOG_INTERVAL = 30.0  # Log progress every 30 seconds
STALL_WARNING_TIME = 60.0  # Warn after 60 seconds of no progress
STALL_ABORT_TIME = 300.0  # Abort after 5 minutes of no progress
MAX_PHASE_TIME = 5400.0  #  Abort after 90 minutes for any single phase


class TemperatureStallError(RuntimeError):
    """Raised when temperature stops making progress toward the target."""


@dataclass(frozen=True)
class TouchRecord:
    """One touch re-reference: where the bed was found, relative to the Z frame before it."""

    time: float
    height: float
    coil_temperature: float
    trigger: float
    cumulative: float  # total Z correction since the calibration's starting home


def _write_samples(samples: list[Sample], label: str) -> list[str]:
    if not samples:
        return []
    path = generate_filepath(label)
    try:
        write_samples_to_csv(samples, path)
    except Exception as e:
        logger.warning("Failed to write samples to CSV: %s", e)
        return []
    logger.info("Wrote raw data to: %s", path)
    return [path]


def _write_touches(touches: list[TouchRecord], path: str) -> None:
    try:
        with open(path, "w", newline="") as f:
            _ = f.write("time,height,coil_temperature,trigger,cumulative\n")
            for t in touches:
                _ = f.write(f"{t.time},{t.height},{t.coil_temperature},{t.trigger},{t.cumulative}\n")
    except Exception as e:
        logger.warning("Failed to write touch log: %s", e)


def collection_targets(start: float, max_temp: int, touch_step: float) -> list[float]:
    """Coil temperatures a heating phase collects through, with a touch re-reference before each."""
    if not touch_step:
        return [max_temp]
    targets: list[float] = []
    target = start + touch_step
    while target < max_temp:
        targets.append(target)
        target += touch_step
    targets.append(max_temp)
    return targets


@dataclass(frozen=True)
class TemperatureCalibrateParams:
    """Parameters for CARTOGRAPHER_CALIBRATE_TEMPERATURE."""

    min_temp: int = param("Minimum coil temperature", default=40, min=40, max=50)
    max_temp: int = param("Maximum coil temperature", default=60, min=60, max=90)
    bed_temp: int = param("Bed temperature target", default=90, min=90, max=120)
    z_speed: int = param("Z movement speed", default=5, min=1)
    touch_step: float = param(
        "Coil temperature rise (C) between touch re-references of Z while heating. 0 disables."
        " Keeps each phase at its true height as the bed and frame grow, so that growth is not"
        " folded into the coil model.",
        default=0,
        min=0,
    )


@final
class TemperatureCalibrateMacro(Macro):
    description = "Calibrate temperature compensation for frequency drift"

    def __init__(
        self,
        mcu: Mcu,
        toolhead: Toolhead,
        config: Configuration,
        gcode: GCodeDispatch,
        task_executor: TaskExecutor,
        scheduler: Scheduler,
        touch: ProbeMode | None = None,
    ) -> None:
        self.touch = touch
        self.mcu = mcu
        self.toolhead = toolhead
        self.config = config
        self.gcode = gcode
        self.task_executor = task_executor
        self.scheduler = scheduler

    @override
    def run(self, params: MacroParams) -> None:
        scipy_helpers.raise_if_curve_fit_unavailable()

        p = parse(TemperatureCalibrateParams, params)

        if p.max_temp < p.min_temp + 20:
            msg = f"MAX_TEMP ({p.max_temp}) must be at least MIN_TEMP + 20 ({p.min_temp + 20})"
            raise RuntimeError(msg)
        if p.bed_temp < p.max_temp:
            msg = f"BED_TEMP ({p.bed_temp}) must be at least MAX_TEMP ({p.max_temp})"
            raise RuntimeError(msg)

        if p.touch_step and self.touch is None:
            msg = "TOUCH_STEP needs touch probing, which is not available"
            raise RuntimeError(msg)
        if not self.toolhead.is_homed("x") or not self.toolhead.is_homed("y") or not self.toolhead.is_homed("z"):
            msg = "Must home axes before temperature calibration"
            raise RuntimeError(msg)

        _, max_z = self.toolhead.get_axis_limits("z")
        cooling_height = max_z * 2 / 3
        logger.info(
            "Starting temperature calibration sequence... (bed=%d°C range=%d-%d°C, cooling height=%.1fmm)",
            p.bed_temp,
            p.min_temp,
            p.max_temp,
            cooling_height,
        )
        self.toolhead.move(z=cooling_height, speed=p.z_speed)
        self.toolhead.move(
            x=self.config.bed_mesh.zero_reference_position[0],
            y=self.config.bed_mesh.zero_reference_position[1],
            speed=self.config.general.travel_speed,
        )

        # Collect data at 3 different heights
        data_per_height: dict[float, list[Sample]] = {}
        heights = [1, 2, 3]
        csv_files: list[str] = []
        touches: list[TouchRecord] = []
        touches_path = generate_filepath("temp_calib_touches") if p.touch_step else None

        # The MCU only streams while a session is open, and callbacks alone do not open one:
        # without this, coil temperature stays frozen at the last sample and every wait
        # stalls. The session never starts collecting, so it keeps nothing in memory.
        with self.mcu.start_session(lambda _: False):
            for phase, height in enumerate(heights, 1):
                logger.info("Starting Phase %d of %d (height=%.1fmm)", phase, len(heights), height)
                cool_samples: list[Sample] = []
                samples: list[Sample] = []
                # Written even if the phase aborts, so hours of data are never lost with it.
                try:
                    self._cool_down_phase(cooling_height, p.min_temp, p.z_speed, cool_samples)
                    self._heat_up_phase(
                        height, p.bed_temp, p.min_temp, p.max_temp, p.z_speed, samples, p.touch_step, touches
                    )
                finally:
                    csv_files += _write_samples(cool_samples, f"temp_calib_cool_before_h{height}mm")
                    csv_files += _write_samples(samples, f"temp_calib_h{height}mm")
                    if touches_path is not None and touches:
                        _write_touches(touches, touches_path)
                data_per_height[height] = samples
                logger.info("Phase %d complete: collected %d samples", phase, len(samples))
        if touches_path is not None:
            csv_files.append(touches_path)

        self.gcode.run_gcode("M140 S0")
        self.toolhead.move(z=cooling_height, speed=p.z_speed)

        model = self.task_executor.run(fit_coil_temperature_model, data_per_height, self.mcu.get_coil_reference())

        self.config.save_coil_model(model)

        logger.info(
            "Temperature calibration complete!\n"
            "The SAVE_CONFIG command will update the printer config file and restart the printer.\n"
            "Raw calibration data can be found in the following files:\n%s",
            "\n".join(csv_files),
        )

    @log_duration("Cooldown phase")
    def _cool_down_phase(self, height: float, min_temp: int, z_speed: int, samples: list[Sample]) -> None:
        """Cool down the probe to minimum temperature, recording free-air samples on the way."""
        logger.info("Cooling probe to %d°C, moving to z %.1f", min_temp, height)

        self.toolhead.move(z=height, speed=z_speed)
        self.toolhead.wait_moves()
        self.gcode.run_gcode("M140 S0\nM106 S255")

        logger.info("Waiting for coil temperature to reach %d°C", min_temp)
        self._collect(samples, target_temp=min_temp, cooling=True)

    @log_duration("Heat up phase")
    def _heat_up_phase(
        self,
        height: float,
        bed_temp: int,
        min_temp: int,
        max_temp: int,
        z_speed: int,
        samples: list[Sample],
        touch_step: float = 0,
        touches: list[TouchRecord] | None = None,
    ) -> None:
        """Heat up and collect samples during temperature rise."""
        logger.info("Starting heaters: bed=%d°C, moving to z %.1f", bed_temp, height)
        self.gcode.run_gcode(f"M140 S{bed_temp}\nM106 S0")

        self.toolhead.move(z=height, speed=z_speed)
        self.toolhead.wait_moves()

        self._wait_for_temperature(target_temp=min_temp - 1, cooling=False)

        logger.info("Collecting data for height %.1f", height)
        phase_start = time.monotonic()
        touching = bool(touch_step)

        for target in collection_targets(min_temp, max_temp, touch_step):
            if touching:
                # Not recording while touching: the coil leaves the phase height.
                touching = self._touch_rereference(height, z_speed, touches)
            self._collect(samples, target_temp=target, cooling=False, phase_start_time=phase_start)
        if touching:
            # Close the last step too, so drift within every step is bracketed.
            _ = self._touch_rereference(height, z_speed, touches)

    def _collect(
        self, samples: list[Sample], target_temp: float, cooling: bool, phase_start_time: float | None = None
    ) -> None:
        """Record samples until the coil reaches target_temp, skipping any from before the last move ended."""
        since = self.toolhead.get_last_move_time()

        def collect(sample: Sample) -> None:
            if sample.time >= since:
                samples.append(sample)

        self.mcu.register_callback(collect)
        try:
            self._wait_for_temperature(target_temp=target_temp, cooling=cooling, phase_start_time=phase_start_time)
        finally:
            self.mcu.unregister_callback(collect)

    def _touch_rereference(self, height: float, z_speed: int, touches: list[TouchRecord] | None) -> bool:
        """
        Touch the bed, reset Z to the measured contact, and return to the phase height.

        Returns False if touching failed twice, so the phase carries on without it
        rather than losing its data.
        """
        assert self.touch is not None
        trigger_pos: float | None = None
        for attempt in (1, 2):
            try:
                trigger_pos = self.touch.perform_probe()
                break
            except (PrinterShutdownError, McuDisconnectedError):
                raise
            except Exception as e:
                logger.warning("Touch re-reference attempt %d failed: %s", attempt, e)

        if trigger_pos is None:
            logger.warning("Continuing this phase without touch re-referencing")
        else:
            pos = self.toolhead.get_position()
            self.toolhead.set_z_position(pos.z - trigger_pos)
            temperature = self._get_current_temperature() or float("nan")
            logger.info(
                "Touch re-reference at coil %.1f°C: bed measured %.4f mm from expected", temperature, trigger_pos
            )
            if touches is not None:
                cumulative = (touches[-1].cumulative if touches else 0.0) + trigger_pos
                touches.append(
                    TouchRecord(self.toolhead.get_last_move_time(), height, temperature, trigger_pos, cumulative)
                )
        self.toolhead.move(z=height, speed=z_speed)
        self.toolhead.wait_moves()
        return trigger_pos is not None

    def _get_current_temperature(self) -> float | None:
        """Get the current coil temperature from the last sample."""
        sample = self.mcu.get_last_sample()
        return sample.temperature if sample is not None else None

    def _wait_for_temperature(self, target_temp: float, cooling: bool, phase_start_time: float | None = None) -> None:
        """
        Wait for coil temperature with progress monitoring.

        Tracks the closest distance to target and detects stalls when
        no progress is made for too long.

        Parameters
        ----------
        target_temp
            The target temperature to reach.
        cooling
            True if waiting for temperature to decrease, False for increase.
        """
        best_remaining: float | None = None
        last_progress_time: float = time.monotonic()
        last_log_time: float = 0.0
        warning_logged = False
        phase = "cool to" if cooling else "heat to"
        if phase_start_time is None:
            phase_start_time = time.monotonic()

        while True:
            self.scheduler.sleep(TEMP_CHECK_INTERVAL)

            current_temp = self._get_current_temperature()
            if current_temp is None:
                continue

            # Check if we've reached the target
            if cooling and current_temp <= target_temp:
                logger.info("Reached target temperature: %.1f°C", current_temp)
                return
            if not cooling and current_temp >= target_temp:
                logger.info("Reached target temperature: %.1f°C", current_temp)
                return

            current_time = time.monotonic()

            elapsed = current_time - phase_start_time
            if elapsed >= MAX_PHASE_TIME:
                action = "cooling" if cooling else "heating"
                msg = (
                    f"Temperature {action} phase exceeded maximum time "
                    f"({MAX_PHASE_TIME / 60:.0f} minutes). "
                    f"Current: {current_temp:.1f}°C, target: {target_temp}°C."
                )
                raise TemperatureStallError(msg)

            remaining = abs(current_temp - target_temp)

            # Check if we made progress (got closer to target)
            if best_remaining is None or remaining < best_remaining:
                best_remaining = remaining
                last_progress_time = current_time
                warning_logged = False

            # Log progress at intervals
            if current_time - last_log_time >= PROGRESS_LOG_INTERVAL:
                logger.info(
                    "Temperature: %.1f°C (%s %d°C, %.1f°C remaining)",
                    current_temp,
                    phase,
                    target_temp,
                    remaining,
                )
                last_log_time = current_time

            # Check for stall (no new progress for too long)
            stall_duration = current_time - last_progress_time
            if stall_duration >= STALL_WARNING_TIME:
                warning_logged = self._handle_stall(
                    stall_duration,
                    current_temp,
                    best_remaining,
                    target_temp,
                    cooling,
                    warning_logged,
                )

    def _handle_stall(
        self,
        stall_duration: float,
        current_temp: float,
        best_remaining: float,
        target_temp: float,
        cooling: bool,
        warning_logged: bool,
    ) -> bool:
        """
        Handle a temperature stall condition.

        Returns whether a warning has been logged.
        """
        action = "cooling" if cooling else "heating"
        if stall_duration >= STALL_ABORT_TIME:
            if cooling:
                suggestion = "If you have an enclosure, try opening the chamber door to improve airflow."
            else:
                suggestion = "If you have an enclosure, try closing the chamber door to retain heat."

            msg = (
                f"Temperature {action} stalled for "
                f"{stall_duration / 60:.0f} minutes: "
                f"stuck at {current_temp:.1f}°C, "
                f"need to reach {target_temp}°C "
                f"({best_remaining:.1f}°C remaining). "
                f"{suggestion}"
            )
            raise TemperatureStallError(msg)

        if not warning_logged:
            if cooling:
                hint = "Consider opening the chamber door if enclosed."
            else:
                hint = "Consider closing the chamber door if enclosed."

            time_until_abort = (STALL_ABORT_TIME - stall_duration) / 60
            logger.warning(
                "Temperature %s appears stalled at %.1f°C "
                "(%.1f°C from target) for %.0fs. %s "
                "Will abort if no progress in %.0f minutes.",
                action,
                current_temp,
                best_remaining,
                stall_duration,
                hint,
                time_until_abort,
            )
            return True

        return warning_logged
