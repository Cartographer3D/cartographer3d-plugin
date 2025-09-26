from __future__ import annotations

import contextlib
import logging
import os
import tempfile
import time
from typing import TYPE_CHECKING, final

import numpy as np
from typing_extensions import override

from cartographer.coil.helpers import line0, line120, line_fit, linear_func
from cartographer.interfaces.configuration import CoilCalibrationConfiguration
from cartographer.interfaces.printer import GCodeDispatch, Macro, MacroParams, Mcu, Sample, Toolhead
from cartographer.lib import scipy_helpers
from cartographer.lib.log import log_duration

if TYPE_CHECKING:
    from cartographer.interfaces.configuration import Configuration

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
    ) -> None:
        self.mcu = mcu
        self.toolhead = toolhead
        self.config = config
        self.gcode = gcode

    @override
    def run(self, params: MacroParams) -> None:
        if not scipy_helpers.is_available():
            msg = "scipy is required for temperature calibration, but is not installed"
            raise RuntimeError(msg)
        min_temp = params.get_int("MIN_TEMP", default=40, minval=40, maxval=50)
        max_temp = params.get_int("MAX_TEMP", default=70, minval=min_temp + 20, maxval=90)
        bed_temp = params.get_int("BED_TEMP", default=90, minval=max_temp, maxval=120)
        z_speed = params.get_int("Z_SPEED", default=5, minval=1)

        if not self.toolhead.is_homed("x") or not self.toolhead.is_homed("y") or not self.toolhead.is_homed("z"):
            msg = "Must home axes before temperature calibration"
            raise RuntimeError(msg)

        logger.info(
            "Starting temperature calibration sequence... (bed=%d°C range=%d-%d°C)",
            bed_temp,
            min_temp,
            max_temp,
        )
        self.toolhead.move(z=10, speed=z_speed)
        self.toolhead.move(
            x=self.config.bed_mesh.zero_reference_position[0],
            y=self.config.bed_mesh.zero_reference_position[1],
            speed=self.config.general.travel_speed,
        )

        # Collect data at 3 different heights
        data_per_height: dict[float, list[Sample]] = {}
        heights = [1.0, 2.0, 3.0]

        for phase, height in enumerate(heights, 1):
            logger.info("Starting Phase %d of %d (height=%.1fmm)", phase, len(heights), height)
            self._cool_down_phase(min_temp, z_speed)
            samples = self._heat_up_phase(height, bed_temp, min_temp, max_temp, z_speed)
            data_per_height[height] = samples

            logger.info("Phase %d complete: collected %d samples", phase, len(samples))
            with contextlib.suppress(Exception):
                self._write_samples_to_csv(samples)

        self.gcode.run_gcode("M140 S0")
        self.toolhead.move(z=50, speed=z_speed)

        model = self._fit_temperature_model(data_per_height)

        self.config.save_coil_model(model)

        logger.info(
            "Temperature calibration complete!\n"
            "The SAVE_CONFIG command will update the printer config file and restart the printer.",
        )

    @log_duration("Cooldown phase")
    def _cool_down_phase(self, min_temp: int, z_speed: int) -> None:
        """Cool down the probe to minimum temperature."""
        logger.info("Cooling probe to %d°C...", min_temp)

        # Move to safe height and turn on cooling
        _, max_z = self.toolhead.get_axis_limits("z")
        self.toolhead.move(z=max_z * 2 / 3, speed=z_speed)
        self.toolhead.wait_moves()
        self.gcode.run_gcode("M140 S0\nM106 S255")

        logger.info("Waiting for coil temperature to reach %d°C", min_temp)
        self.gcode.run_gcode(f"TEMPERATURE_WAIT SENSOR='temperature_sensor {self.config.coil.name}' MAXIMUM={min_temp}")

    @log_duration("Heat up phase")
    def _heat_up_phase(self, height: float, bed_temp: int, min_temp: int, max_temp: int, z_speed: int) -> list[Sample]:
        """Heat up and collect samples during temperature rise."""
        logger.info("Starting heaters: bed=%d°C", bed_temp)
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
                logger.debug("Collected %d samples", count)

        self.mcu.register_callback(callback)
        self.gcode.run_gcode(f"TEMPERATURE_WAIT SENSOR='temperature_sensor {self.config.coil.name}' MINIMUM={max_temp}")
        self.mcu.unregister_callback(callback)

        return samples

    def _fit_temperature_model(self, data_per_height: dict[float, list[Sample]]) -> CoilCalibrationConfiguration:
        """Fit temperature compensation model from collected data."""
        logger.info("Fitting temperature compensation model...")

        # Process each dataset
        a_values: list[float] = []
        b_values: list[float] = []
        freq_values: list[float] = []

        for height, samples in data_per_height.items():
            a, b, freq_at_vertex = process_data(samples)
            a_values.append(a)
            b_values.append(b)
            freq_values.append(freq_at_vertex)
            logger.debug("Height %.1f: a=%.6f, b=%.6f, freq=%.1f", height, a, b, freq_at_vertex)

        fmin = float(np.mean(freq_values))
        freq_offsets = np.array(freq_values) - fmin

        try:
            # Fit a parameters vs frequency offset
            a_params, _ = scipy_helpers.curve_fit(linear_func, freq_offsets, a_values, maxfev=100000)
            a_a, a_b = float(a_params[0]), float(a_params[1])

            # Fit b parameters vs frequency offset
            b_params, _ = scipy_helpers.curve_fit(linear_func, freq_offsets, b_values, maxfev=100000)
            b_a, b_b = float(b_params[0]), float(b_params[1])

            return CoilCalibrationConfiguration(a_a, a_b, b_a, b_b)

        except Exception as e:
            msg = f"Failed to fit temperature compensation parameters: {e}"
            raise RuntimeError(msg) from e

    def _write_samples_to_csv(self, samples: list[Sample]) -> None:
        """Write all samples to CSV file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"cartographer_tempcalib_{timestamp}.csv"

        temp_dir = tempfile.gettempdir()
        output_file = os.path.join(temp_dir, filename)

        with open(output_file, "w", newline="") as f:
            # Write CSV header
            _ = f.write("time,frequency,temperature,position_x,position_y,position_z\n")

            # Write all sample rows
            for sample in samples:
                pos_x = sample.position.x if sample.position else ""
                pos_y = sample.position.y if sample.position else ""
                pos_z = sample.position.z if sample.position else ""

                row = f"{sample.time},{sample.frequency},{sample.temperature},{pos_x},{pos_y},{pos_z}\n"
                _ = f.write(row)


def process_data(samples: list[Sample]) -> tuple[float, float, float]:
    """Process sample data and fit quadratic model."""
    if len(samples) < 1000:
        msg = f"Insufficient samples for calibration: {len(samples)} (need at least 1000)"
        raise RuntimeError(msg)

    dv = int(len(samples) / 1000)
    if dv > 1:
        samples = samples[::dv]

    temps = np.array([s.temperature for s in samples])
    freqs = np.array([s.frequency for s in samples])

    try:
        # Fit quadratic with bounds
        param_bounds = ([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf])
        linear_params, _ = scipy_helpers.curve_fit(
            line_fit, temps, freqs, bounds=param_bounds, maxfev=100000, ftol=1e-10, xtol=1e-10
        )

        # Check if axis is outside reasonable bounds
        axis = -linear_params[1] / (2 * linear_params[0]) if linear_params[0] != 0 else 0

        if axis > 120:
            # Use constrained fit for high axis values
            linear_params1, _ = scipy_helpers.curve_fit(
                line120, temps, freqs, bounds=([0, -np.inf], [np.inf, np.inf]), maxfev=100000, ftol=1e-10, xtol=1e-10
            )
            return (
                linear_params1[0],
                -240 * linear_params1[0],
                line120(np.array([120]), linear_params1[0], linear_params1[1])[0],
            )

        elif axis < 0:
            # Use constrained fit for negative axis values
            linear_params1, _ = scipy_helpers.curve_fit(
                line0, temps, freqs, bounds=([0, -np.inf], [np.inf, np.inf]), maxfev=100000, ftol=1e-10, xtol=1e-10
            )
            return (linear_params1[0], 0, line0(np.array([0]), linear_params1[0], linear_params1[1])[0])

        # Normal case - adjust c parameter to vertex value
        linear_params[2] = line_fit(np.array([axis]), linear_params[0], linear_params[1], linear_params[2])[0]
        return tuple(linear_params)

    except Exception as e:
        msg = f"Failed to fit temperature model: {e}"
        raise RuntimeError(msg) from e
