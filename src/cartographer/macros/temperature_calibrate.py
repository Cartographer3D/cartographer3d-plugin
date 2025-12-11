from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, final

from typing_extensions import override

from cartographer.coil.calibration import fit_coil_temperature_model
from cartographer.interfaces.printer import GCodeDispatch, Macro, MacroParams, Mcu, Sample, Toolhead
from cartographer.lib import scipy_helpers
from cartographer.lib.csv import generate_filepath, write_samples_to_csv
from cartographer.lib.log import log_duration

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
    ) -> None:
        self.mcu = mcu
        self.toolhead = toolhead
        self.config = config
        self.gcode = gcode
        self.task_executor = task_executor
        self.scheduler = scheduler

    @override
    def run(self, params: MacroParams) -> None:
        if not scipy_helpers.is_available():
            msg = "scipy is required for temperature calibration, but is not installed"
            raise RuntimeError(msg)

        min_temp = params.get_int("MIN_TEMP", default=40, minval=40, maxval=50)
        max_temp = params.get_int("MAX_TEMP", default=60, minval=min_temp + 20, maxval=90)
        bed_temp = params.get_int("BED_TEMP", default=90, minval=max_temp + 30, maxval=120)
        z_speed = params.get_int("Z_SPEED", default=5, minval=1)

        if not self.toolhead.is_homed("x") or not self.toolhead.is_homed("y") or not self.toolhead.is_homed("z"):
            msg = "Must home axes before temperature calibration"
            raise RuntimeError(msg)

        _, max_z = self.toolhead.get_axis_limits("z")
        cooling_height = max_z * 2 / 3
        logger.info(
            "Starting temperature calibration sequence... (bed=%d°C range=%d-%d°C, cooling height=%.1fmm)",
            bed_temp,
            min_temp,
            max_temp,
            cooling_height,
        )
        self.toolhead.move(z=cooling_height, speed=z_speed)
        self.toolhead.move(
            x=self.config.bed_mesh.zero_reference_position[0],
            y=self.config.bed_mesh.zero_reference_position[1],
            speed=self.config.general.travel_speed,
        )

        # Collect data at 3 different heights
        data_per_height: dict[float, list[Sample]] = {}
        heights = [1, 2, 3]
        csv_files: list[str] = []

        for phase, height in enumerate(heights, 1):
            logger.info("Starting Phase %d of %d (height=%.1fmm)", phase, len(heights), height)
            self._cool_down_phase(cooling_height, min_temp, z_speed)
            samples = self._heat_up_phase(height, bed_temp, min_temp, max_temp, z_speed)
            data_per_height[height] = samples

            logger.info("Phase %d complete: collected %d samples", phase, len(samples))
            path = generate_filepath(f"temp_calib_h{height}mm")
            try:
                write_samples_to_csv(samples, path)
                logger.info("Wrote raw data to: %s", path)
                csv_files.append(path)
            except Exception as e:
                logger.warning("Failed to write samples to CSV: %s", e)

        self.gcode.run_gcode("M140 S0")
        self.toolhead.move(z=cooling_height, speed=z_speed)

        model = self.task_executor.run(fit_coil_temperature_model, data_per_height, self.mcu.get_coil_reference())

        self.config.save_coil_model(model)

        logger.info(
            "Temperature calibration complete!\n"
            "The SAVE_CONFIG command will update the printer config file and restart the printer.\n"
            "Raw calibration data can be found in the following files:\n%s",
            "\n".join(csv_files),
        )

    @log_duration("Cooldown phase")
    def _cool_down_phase(self, height: float, min_temp: int, z_speed: int) -> None:
        """Cool down the probe to minimum temperature."""
        logger.info("Cooling probe to %d°C, moving to z %.1f", min_temp, height)

        self.toolhead.move(z=height, speed=z_speed)
        self.toolhead.wait_moves()
        self.gcode.run_gcode("M140 S0\nM106 S255")

        logger.info("Waiting for coil temperature to reach %d°C", min_temp)
        self._wait_for_temperature(target_temp=min_temp, cooling=True)

    @log_duration("Heat up phase")
    def _heat_up_phase(self, height: float, bed_temp: int, min_temp: int, max_temp: int, z_speed: int) -> list[Sample]:
        """Heat up and collect samples during temperature rise."""
        logger.info("Starting heaters: bed=%d°C, moving to z %.1f", bed_temp, height)
        self.gcode.run_gcode(f"M140 S{bed_temp}\nM106 S0")

        self.toolhead.move(z=height, speed=z_speed)
        self.toolhead.wait_moves()

        self._wait_for_temperature(target_temp=min_temp - 1, cooling=False)

        logger.info("Collecting data for height %.1f", height)
        samples: list[Sample] = []

        self.mcu.register_callback(samples.append)
        try:
            self._wait_for_temperature(target_temp=max_temp, cooling=False)
        finally:
            self.mcu.unregister_callback(samples.append)

        return samples

    def _get_current_temperature(self) -> float | None:
        """Get the current coil temperature from the last sample."""
        sample = self.mcu.get_last_sample()
        return sample.temperature if sample is not None else None

    def _wait_for_temperature(self, target_temp: int, cooling: bool) -> None:
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
        phase_start_time: float = time.monotonic()

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
        target_temp: int,
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
