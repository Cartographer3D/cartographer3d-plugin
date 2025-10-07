from __future__ import annotations

import logging
from typing import TYPE_CHECKING, final

from typing_extensions import override

from cartographer.coil.calibration import fit_coil_temperature_model
from cartographer.interfaces.printer import GCodeDispatch, Macro, MacroParams, Mcu, Sample, Toolhead
from cartographer.lib import scipy_helpers
from cartographer.lib.csv import generate_filepath, write_samples_to_csv
from cartographer.lib.log import log_duration

if TYPE_CHECKING:
    from cartographer.interfaces.configuration import Configuration
    from cartographer.interfaces.multiprocessing import TaskExecutor

logger = logging.getLogger(__name__)


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
    ) -> None:
        self.mcu = mcu
        self.toolhead = toolhead
        self.config = config
        self.gcode = gcode
        self.task_executor = task_executor

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

        # Move to safe height and turn on cooling
        self.toolhead.move(z=height, speed=z_speed)
        self.toolhead.wait_moves()
        self.gcode.run_gcode("M140 S0\nM106 S255")

        logger.info("Waiting for coil temperature to reach %d°C", min_temp)
        self.gcode.run_gcode(f"TEMPERATURE_WAIT SENSOR='temperature_sensor {self.config.coil.name}' MAXIMUM={min_temp}")

    @log_duration("Heat up phase")
    def _heat_up_phase(self, height: float, bed_temp: int, min_temp: int, max_temp: int, z_speed: int) -> list[Sample]:
        """Heat up and collect samples during temperature rise."""
        logger.info("Starting heaters: bed=%d°C, moving to z %.1f", bed_temp, height)
        self.gcode.run_gcode(f"M140 S{bed_temp}\nM106 S0")

        self.toolhead.move(z=height, speed=z_speed)
        self.toolhead.wait_moves()

        self.gcode.run_gcode(
            f"TEMPERATURE_WAIT SENSOR='temperature_sensor {self.config.coil.name}' MINIMUM={min_temp - 1}"
        )
        logger.info("Collecting data for height %.1f", height)
        samples: list[Sample] = []

        def callback(sample: Sample):
            nonlocal samples
            samples.append(sample)
            count = len(samples)
            if count > 0 and count % 100 == 0:
                logger.info("Collected %d samples", count)

        self.mcu.register_callback(callback)
        self.gcode.run_gcode(f"TEMPERATURE_WAIT SENSOR='temperature_sensor {self.config.coil.name}' MINIMUM={max_temp}")
        self.mcu.unregister_callback(callback)

        return samples
