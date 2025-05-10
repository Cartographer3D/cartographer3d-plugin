from __future__ import annotations

from typing import TYPE_CHECKING, cast

from numpy.polynomial import Polynomial
from typing_extensions import override

from cartographer.interfaces.configuration import ScanModelConfiguration, ScanModelFit
from cartographer.probe.model import ModelSelectorMixin

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cartographer.interfaces.printer import Sample


MAX_TOLERANCE = 1e-8
ITERATIONS = 50
DEGREES = 9


# TODO: Temperature compensation
class ScanModel:
    _poly: Polynomial | None = None

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def z_offset(self) -> float:
        return self._config.z_offset

    @property
    def poly(self) -> Polynomial:
        if self._poly is None:
            self._poly = Polynomial(self._config.coefficients, self._config.domain)
        return self._poly

    def __init__(self, config: ScanModelConfiguration) -> None:
        self._config: ScanModelConfiguration = config

    @staticmethod
    def fit(samples: Sequence[Sample]) -> ScanModelFit:
        positions = [sample.position for sample in samples]
        # TODO: Can we ignore missing positions?
        if not all(positions):
            msg = "not all samples are valid, try again"
            raise RuntimeError(msg)
        z_offsets = [pos.z for pos in positions if pos is not None]
        inverse_frequencies = [1 / sample.frequency for sample in samples]

        poly = cast("Polynomial", Polynomial.fit(inverse_frequencies, z_offsets, DEGREES))

        return ScanModelFit(
            coefficients=poly.coef,
            domain=poly.domain,
        )

    def frequency_to_distance(self, frequency: float) -> float:
        lower_bound, upper_bound = self._config.domain
        inverse_frequency = 1 / frequency

        if inverse_frequency > upper_bound:
            return float("inf")
        elif inverse_frequency < lower_bound:
            return float("-inf")

        return self._eval(inverse_frequency) + self._config.z_offset

    def distance_to_frequency(self, distance: float) -> float:
        # PERF: We can use brentq if scipy is available
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html#scipy.optimize.brentq
        distance -= self._config.z_offset
        min_z, max_z = self._get_z_range()
        if distance < min_z or distance > max_z:
            msg = f"attempted to map out-of-range distance {distance:.3f}, valid range [{min_z:.3f}, {max_z:.3f}]"
            raise RuntimeError(msg)

        lower_bound, upper_bound = self._config.domain

        for _ in range(ITERATIONS):
            midpoint = (upper_bound + lower_bound) / 2
            value = self._eval(midpoint)

            if abs(value - distance) < MAX_TOLERANCE:
                return float(1.0 / midpoint)
            elif value < distance:
                lower_bound = midpoint
            else:
                upper_bound = midpoint

        msg = "model convergence error"
        raise RuntimeError(msg)

    _z_range: tuple[float, float] | None = None

    def _get_z_range(self) -> tuple[float, float]:
        if self._z_range is None:
            min, max = self._config.domain
            self._z_range = (self._eval(min), self._eval(max))
        return self._z_range

    def _eval(self, x: float) -> float:
        return float(self.poly(x))  # pyright: ignore[reportUnknownArgumentType]


class ScanModelSelectorMixin(ModelSelectorMixin[ScanModel, ScanModelConfiguration]):
    @override
    def _create_model(self, config: ScanModelConfiguration) -> ScanModel:
        return ScanModel(config)
