from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
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
    from collections.abc import Callable, Sequence

    from cartographer.interfaces.configuration import Configuration
    from cartographer.interfaces.multiprocessing import Scheduler, TaskExecutor
    from cartographer.probe.scan_mode import ScanMode

logger = logging.getLogger(__name__)

# Temperature monitoring constants
TEMP_CHECK_INTERVAL = 1.0  # Check temperature every second
PROGRESS_LOG_INTERVAL = 30.0  # Log progress every 30 seconds
STALL_WARNING_TIME = 60.0  # Warn after 60 seconds of no progress
STALL_ABORT_TIME = 300.0  # Abort after 5 minutes of no progress
MAX_PHASE_TIME = 5400.0  #  Abort after 90 minutes for any single phase
# Keep one sample per interval. Coil temperature moves over minutes, so ~10 Hz is ample
# for the fit, whereas keeping every sample (~600 Hz) piles up enough objects over a
# phase that Klipper's garbage collection blocked the reactor for 0.38 s and the main
# MCU shut down with "Timer too close".
SAMPLE_INTERVAL = 0.1
# Interleaved mode: samples within this long after a move are dropped (coil settling).
SETTLE_TIME = 0.2


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


class _HeatProgress:
    """Stall and timeout guard for the interleaved ramp, with the same limits as a phase."""

    def __init__(self) -> None:
        self._start: float = time.monotonic()
        self._best: float | None = None
        self._last_progress: float = self._start
        self._last_log: float = 0.0

    def update(self, temperature: float, target: float) -> None:
        now = time.monotonic()
        if self._best is None or temperature > self._best:
            self._best = temperature
            self._last_progress = now
        if now - self._last_log >= PROGRESS_LOG_INTERVAL:
            logger.info("Temperature: %.1f°C (heat to %.0f°C, interleaved)", temperature, target)
            self._last_log = now
        if now - self._last_progress >= STALL_ABORT_TIME:
            msg = f"Coil temperature stalled at {temperature:.1f}°C (target {target:.0f}°C)"
            raise TemperatureStallError(msg)
        if now - self._start >= MAX_PHASE_TIME:
            msg = f"Interleaved heating exceeded {MAX_PHASE_TIME / 60:.0f} minutes at {temperature:.1f}°C"
            raise TemperatureStallError(msg)


def correct_for_growth(
    samples: list[Sample], height: float, touches: Sequence[TouchRecord], freq_at: Callable[[float], float]
) -> list[Sample]:
    """
    Shift each sample's frequency to what it would read at the nominal `height`.

    Each touch reset Z to the contact, so touch k's trigger is how far the bed moved since the
    previous re-reference. Between touches the gap is taken as `height - trigger * elapsed/interval`
    (growth accruing linearly in time), which removes the sawtooth left by growth between touches.
    Only touches after the first sample are used, so other phases' touches don't apply.
    """
    if not samples:
        return samples
    start = samples[0].time
    relevant = [t for t in touches if t.time > start]
    if not relevant:
        return samples
    edges = [start] + [t.time for t in relevant]
    nominal = freq_at(height)
    corrected: list[Sample] = []
    k = 1
    for sample in samples:
        while k < len(edges) - 1 and sample.time > edges[k]:
            k += 1
        fraction = min(max((sample.time - edges[k - 1]) / (edges[k] - edges[k - 1]), 0.0), 1.0)
        gap = height - relevant[k - 1].trigger * fraction
        corrected.append(replace(sample, frequency=sample.frequency + nominal - freq_at(gap)))
    return corrected


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
    interleave: bool = param(
        "Cycle through all heights during ONE heating ramp instead of cooling and reheating"
        " once per height (0). Several times faster, and every height sees the same thermal state.",
        default=True,
    )
    dwell: float = param("Seconds at each height per cycle when interleaving", default=3.0, min=0.5)


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
        scan: ScanMode | None = None,
    ) -> None:
        self.touch = touch
        self.scan = scan
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
            if p.interleave:
                data_per_height = self._run_interleaved(heights, cooling_height, p, touches, touches_path, csv_files)
            else:
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

        if touches:
            data_per_height = self._correct_for_growth(data_per_height, touches)
        model = self.task_executor.run(fit_coil_temperature_model, data_per_height, self.mcu.get_coil_reference())

        self.config.save_coil_model(model)

        logger.info(
            "Temperature calibration complete!\n"
            "The SAVE_CONFIG command will update the printer config file and restart the printer.\n"
            "Raw calibration data can be found in the following files:\n%s",
            "\n".join(csv_files),
        )

    def _run_interleaved(
        self,
        heights: Sequence[float],
        cooling_height: float,
        p: TemperatureCalibrateParams,
        touches: list[TouchRecord],
        touches_path: str | None,
        csv_files: list[str],
    ) -> dict[float, list[Sample]]:
        """One cooldown, then one heating ramp that cycles through every height."""
        cool_samples: list[Sample] = []
        per_height: dict[float, list[Sample]] = {h: [] for h in heights}
        try:
            self._cool_down_phase(cooling_height, p.min_temp, p.z_speed, cool_samples)
            self._heat_up_interleaved(heights, p, per_height, touches)
        finally:
            csv_files += _write_samples(cool_samples, "temp_calib_cool_before")
            for h in heights:
                csv_files += _write_samples(per_height[h], f"temp_calib_h{h}mm")
            if touches_path is not None and touches:
                _write_touches(touches, touches_path)
        for h in heights:
            logger.info("Height %.1fmm: collected %d samples", h, len(per_height[h]))
        return per_height

    @log_duration("Interleaved heat up")
    def _heat_up_interleaved(
        self,
        heights: Sequence[float],
        p: TemperatureCalibrateParams,
        per_height: dict[float, list[Sample]],
        touches: list[TouchRecord],
    ) -> None:
        logger.info("Starting heaters: bed=%d°C, cycling heights %s", p.bed_temp, heights)
        self.gcode.run_gcode(f"M140 S{p.bed_temp}\nM106 S0")
        self.toolhead.move(z=heights[0], speed=p.z_speed)
        self.toolhead.wait_moves()
        self._wait_for_temperature(target_temp=p.min_temp - 1, cooling=False)

        progress = _HeatProgress()
        touching = bool(p.touch_step)
        for target in collection_targets(p.min_temp, p.max_temp, p.touch_step):
            if touching:
                touching = self._touch_rereference(heights[0], p.z_speed, touches)
            reached = False
            while not reached:
                for height in heights:
                    self.toolhead.move(z=height, speed=p.z_speed)
                    self.toolhead.wait_moves()
                    reached = self._dwell(per_height[height], p.dwell, target, progress)
                    if reached:
                        break
        if touching:
            _ = self._touch_rereference(heights[0], p.z_speed, touches)

    def _dwell(self, samples: list[Sample], seconds: float, target: float, progress: _HeatProgress) -> bool:
        """Collect at the current height for up to `seconds`; True once the coil reaches `target`."""
        next_time = self.toolhead.get_last_move_time() + SETTLE_TIME

        def collect(sample: Sample) -> None:
            nonlocal next_time
            if sample.time >= next_time:
                samples.append(sample)
                next_time = sample.time + SAMPLE_INTERVAL

        self.mcu.register_callback(collect)
        try:
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                self.scheduler.sleep(SAMPLE_INTERVAL)
                temperature = self._get_current_temperature()
                if temperature is None:
                    continue
                progress.update(temperature, target)
                if temperature >= target:
                    return True
        finally:
            self.mcu.unregister_callback(collect)
        return False

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
        """
        Record samples until the coil reaches target_temp.

        Skips samples from before the last move ended, and keeps one per SAMPLE_INTERVAL.
        """
        next_time = self.toolhead.get_last_move_time()

        def collect(sample: Sample) -> None:
            nonlocal next_time
            if sample.time >= next_time:
                samples.append(sample)
                next_time = sample.time + SAMPLE_INTERVAL

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

    def _correct_for_growth(
        self, data_per_height: dict[float, list[Sample]], touches: list[TouchRecord]
    ) -> dict[float, list[Sample]]:
        if self.scan is None or not self.scan.has_model():
            logger.warning("No scan model loaded: fitting without correcting for growth between touches")
            return data_per_height
        model = self.scan.get_model()
        reference = model.config.reference_temperature

        def freq_at(distance: float) -> float:
            # At the model's reference temperature, so no coil compensation is applied.
            return model.distance_to_frequency(distance, temperature=reference)

        logger.info(
            "Correcting %d touch intervals for bed growth (%.4f mm total)", len(touches), touches[-1].cumulative
        )
        return {h: correct_for_growth(samples, h, touches, freq_at) for h, samples in data_per_height.items()}

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
