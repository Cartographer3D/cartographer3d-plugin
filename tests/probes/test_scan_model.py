from __future__ import annotations

import math
from typing import Callable, cast

import pytest
from numpy.polynomial import Polynomial
from typing_extensions import TypeAlias

from cartographer.interfaces.configuration import ScanModelConfiguration
from cartographer.interfaces.printer import Position, Sample
from cartographer.probe.scan_model import ScanModel

ScanModelFactory: TypeAlias = Callable[[float], ScanModel]


@pytest.fixture
def model_factory() -> ScanModelFactory:
    def factory(z_offset: float) -> ScanModel:
        poly = Polynomial([0, 1])
        poly = cast("Polynomial", poly.convert(domain=[1 / 5.5, 10]))
        config = ScanModelConfiguration("test", poly.coef, poly.domain, z_offset)

        model = ScanModel(config)
        return model

    return factory


def test_fit() -> None:
    samples = [Sample(time=i, frequency=1 / i, position=Position(0, 0, 0), temperature=0) for i in range(1, 20)]

    fit = ScanModel.fit("test", samples, 0)

    assert fit.domain[0] == 1
    assert fit.domain[1] == 19


def test_frequency_to_distance(model_factory: ScanModelFactory) -> None:
    model = model_factory(0.0)
    frequency = 3.0
    distance = model.frequency_to_distance(frequency)
    assert isinstance(distance, float)
    assert distance != math.inf and distance != -math.inf


def test_distance_to_frequency(model_factory: ScanModelFactory) -> None:
    model = model_factory(0.0)
    distance = 2.5
    frequency = model.distance_to_frequency(distance)
    assert isinstance(frequency, float)
    assert frequency > 0


def test_distance_to_frequency_out_of_range(model_factory: ScanModelFactory) -> None:
    model = model_factory(0)
    with pytest.raises(RuntimeError, match="Attempted to map out-of-range distance"):
        _ = model.distance_to_frequency(11)  # Out of z_range


def test_frequency_to_distance_applies_offset(model_factory: ScanModelFactory) -> None:
    model = model_factory(-0.5)
    frequency = 1 / 3.0

    distance = model.frequency_to_distance(frequency)

    assert distance == 2.5


def test_distance_to_frequency_applies_offset(model_factory: ScanModelFactory) -> None:
    model = model_factory(-0.5)
    distance = 2.5

    frequency = model.distance_to_frequency(distance)

    assert frequency == pytest.approx(1 / 3)  # pyright: ignore[reportUnknownMemberType]


def test_frequency_to_distance_out_of_range(model_factory: ScanModelFactory) -> None:
    model = model_factory(0)
    low_frequency_dist = model.frequency_to_distance(1 / 500)  # Out of z_range
    high_frequency_dist = model.frequency_to_distance(1000000)  # Out of z_range

    assert low_frequency_dist == float("inf")
    assert high_frequency_dist == float("-inf")
