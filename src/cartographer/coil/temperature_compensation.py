from __future__ import annotations

import logging
from functools import cached_property
from typing import TYPE_CHECKING, Protocol, final

import numpy as np
from typing_extensions import override

from cartographer.coil.helpers import param_linear
from cartographer.probe.scan_model import TemperatureCompensationModel

if TYPE_CHECKING:
    from cartographer.interfaces.configuration import CoilCalibrationConfiguration
    from cartographer.interfaces.printer import CoilCalibrationReference


logger = logging.getLogger(__name__)


class CoilReferenceMcu(Protocol):
    def get_coil_reference(self) -> CoilCalibrationReference: ...


@final
class CoilTemperatureCompensationModel(TemperatureCompensationModel):
    @cached_property
    def coil_reference(self) -> CoilCalibrationReference:
        return self._mcu.get_coil_reference()

    def __init__(self, config: CoilCalibrationConfiguration, mcu: CoilReferenceMcu) -> None:
        self._mcu = mcu
        self.a_a = config.a_a
        self.a_b = config.a_b
        self.b_a = config.b_a
        self.b_b = config.b_b

    @override
    def compensate(self, frequency: float, temp_source: float, temp_target: float) -> float:
        """
        Compensate frequency based on temperature change using quadratic parameter interpolation.
        """
        # TODO: Should this be compensated to the target temperature?
        ref_freq = self.coil_reference.min_frequency
        freq_offset = frequency - ref_freq
        param_a_interp, param_b_interp = self._interpolate_parameters(freq_offset)

        quad_coeffs = self._build_quadratic_coefficients(temp_source, freq_offset)
        discriminant = self._calculate_discriminant(quad_coeffs)

        if discriminant < 0:
            return self._apply_linear_compensation(frequency, temp_source, temp_target, param_a_interp, param_b_interp)
        else:
            return self._apply_quadratic_compensation(temp_target, discriminant, quad_coeffs)

    def _interpolate_parameters(self, freq_offset: float) -> tuple[float, float]:
        """Calculate interpolated parameters for the given frequency offset."""
        param_a_interp = param_linear(freq_offset, self.a_a, self.a_b)
        param_b_interp = param_linear(freq_offset, self.b_a, self.b_b)
        return param_a_interp, param_b_interp

    def _build_quadratic_coefficients(self, temp: float, freq_offset: float) -> tuple[float, float, float]:
        """Build coefficients for the quadratic equation ax² + bx + c = 0."""
        temp_sq = temp**2

        quad_a = 4 * (temp * self.a_a) ** 2 + 4 * temp * self.a_a * self.b_a + self.b_a**2 + 4 * self.a_a

        quad_b = (
            8 * temp_sq * self.a_a * self.a_b
            + 4 * temp * (self.a_a * self.b_b + self.a_b * self.b_a)
            + 2 * self.b_a * self.b_b
            + 4 * self.a_b
            - 4 * freq_offset * self.a_a
        )

        quad_c = 4 * (temp * self.a_b) ** 2 + 4 * temp * self.a_b * self.b_b + self.b_b**2 - 4 * freq_offset * self.a_b

        return quad_a, quad_b, quad_c

    def _calculate_discriminant(self, quad_coeffs: tuple[float, float, float]) -> float:
        """Calculate the discriminant of the quadratic equation."""
        quad_a, quad_b, quad_c = quad_coeffs
        return quad_b**2 - 4 * quad_a * quad_c

    def _apply_linear_compensation(
        self, freq: float, temp_source: float, temp_target: float, param_a_interp: float, param_b_interp: float
    ) -> float:
        """Apply linear compensation when quadratic solution is not real."""
        # Calculate the constant term by removing temperature-dependent parts from original frequency
        param_c = freq - param_a_interp * temp_source**2 - param_b_interp * temp_source

        # Apply compensation to target temperature
        return param_a_interp * temp_target**2 + param_b_interp * temp_target + param_c

    def _apply_quadratic_compensation(
        self, temp_target: float, discriminant: float, quad_coeffs: tuple[float, float, float]
    ) -> float:
        """Apply quadratic compensation using the discriminant solution."""
        quad_a, quad_b, _ = quad_coeffs
        ref_freq = self.coil_reference.min_frequency

        # Solve quadratic equation
        ax = (np.sqrt(discriminant) - quad_b) / (2 * quad_a)

        # Get parameters at the solution point
        param_a = param_linear(ax, self.a_a, self.a_b)
        param_b = param_linear(ax, self.b_a, self.b_b)

        # Apply compensation with safeguard against division by zero
        if abs(param_a) > 1e-12:  # More robust zero check
            temp_offset = param_b / (2 * param_a)
            return param_a * (temp_target + temp_offset) ** 2 + ax + ref_freq
        else:
            # Linear fallback when param_a is effectively zero
            return param_b * temp_target + ax + ref_freq
