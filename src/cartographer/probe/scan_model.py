from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
from numpy.polynomial import Polynomial
from typing_extensions import override

from cartographer.interfaces.configuration import ScanModelConfiguration
from cartographer.probe.model import ModelSelectorMixin

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cartographer.interfaces.printer import Sample


MAX_TOLERANCE = 1e-8
ITERATIONS = 50
DEGREES = 9


class TemperatureCompensationModel(Protocol):
    def compensate(self, frequency: float, temp_source: float, temp_target: float) -> float: ...


class _NoTemperatureCompensationModel(TemperatureCompensationModel):
    @override
    def compensate(self, frequency: float, temp_source: float, temp_target: float) -> float:
        return frequency


class ScanModel:
    @property
    def name(self) -> str:
        return self.config.name

    @property
    def z_offset(self) -> float:
        return self.config.z_offset

    @cached_property
    def _poly(self) -> Polynomial:
        return Polynomial(self.config.coefficients, domain=self.config.domain)

    def __init__(
        self, config: ScanModelConfiguration, temperature_compensation: TemperatureCompensationModel | None
    ) -> None:
        self.config: ScanModelConfiguration = config
        self.temperature_compensation: TemperatureCompensationModel = (
            temperature_compensation or _NoTemperatureCompensationModel()
        )

    @staticmethod
    def fit(name: str, samples: Sequence[Sample], z_offset: float) -> ScanModelConfiguration:
        positions = [sample.position for sample in samples]
        # TODO: Can we ignore missing positions?
        if not all(positions):
            msg = "Not all samples are valid, try again"
            raise RuntimeError(msg)
        z_offsets = [pos.z for pos in positions if pos is not None]
        inverse_frequencies = [1 / sample.frequency for sample in samples]

        poly = cast("Polynomial", Polynomial.fit(inverse_frequencies, z_offsets, DEGREES))
        converted = cast("Polynomial", poly.convert(domain=poly.domain))
        temperature = float(np.mean([sample.temperature for sample in samples]))

        return ScanModelConfiguration(
            name=name,
            coefficients=converted.coef,
            domain=converted.domain,
            z_offset=z_offset,
            reference_temperature=temperature,
        )

    def frequency_to_distance(self, frequency: float, *, temperature: float) -> float:
        return self._raw_frequency_to_distance(
            self.temperature_compensation.compensate(
                frequency,
                temp_source=temperature,
                temp_target=self.config.reference_temperature,
            )
        )

    def _raw_frequency_to_distance(self, frequency: float) -> float:
        lower_bound, upper_bound = self.config.domain
        inverse_frequency = 1 / frequency

        if inverse_frequency > upper_bound:
            return float("inf")
        elif inverse_frequency < lower_bound:
            return float("-inf")

        return self._eval(inverse_frequency) + self.config.z_offset

    def distance_to_frequency(self, distance: float, *, temperature: float) -> float:
        frequency = self._distance_to_raw_frequency(distance)

        return self.temperature_compensation.compensate(
            frequency,
            temp_source=self.config.reference_temperature,
            temp_target=temperature,
        )

    def _distance_to_raw_frequency(self, distance: float) -> float:
        # PERF: We can use brentq if scipy is available
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html#scipy.optimize.brentq
        distance -= self.config.z_offset
        min_z, max_z = self._get_z_range()
        if distance < min_z or distance > max_z:
            msg = f"Attempted to map out-of-range distance {distance:.3f}, valid range [{min_z:.3f}, {max_z:.3f}]"
            raise RuntimeError(msg)

        lower_bound, upper_bound = self.config.domain

        for _ in range(ITERATIONS):
            midpoint = (upper_bound + lower_bound) / 2
            value = self._eval(midpoint)

            if abs(value - distance) < MAX_TOLERANCE:
                return float(1.0 / midpoint)
            elif value < distance:
                lower_bound = midpoint
            else:
                upper_bound = midpoint

        msg = "Model convergence error"
        raise RuntimeError(msg)

    _z_range: tuple[float, float] | None = None

    def _get_z_range(self) -> tuple[float, float]:
        if self._z_range is None:
            min, max = self.config.domain
            self._z_range = (self._eval(min), self._eval(max))
        return self._z_range

    def _eval(self, x: float) -> float:
        return float(self._poly(x))  # pyright: ignore[reportUnknownArgumentType]


class ScanModelSelectorMixin(ModelSelectorMixin[ScanModel, ScanModelConfiguration], ABC):
    @abstractmethod
    def get_compensation_model(self) -> TemperatureCompensationModel | None: ...

    @override
    def _create_model(self, config: ScanModelConfiguration) -> ScanModel:
        return ScanModel(config, self.get_compensation_model())
