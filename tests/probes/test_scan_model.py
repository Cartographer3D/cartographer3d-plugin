from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Union, cast

import pytest
from numpy.polynomial import Polynomial
from pytest import approx  # pyright: ignore[reportUnknownVariableType]
from typing_extensions import TypeAlias

from cartographer.interfaces.configuration import ScanModelConfiguration
from cartographer.interfaces.printer import Position, Sample
from cartographer.probe.scan_model import ScanModel, TemperatureCompensationModel

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

ScanModelFactory: TypeAlias = Callable[[float, Union[TemperatureCompensationModel, None]], ScanModel]


@pytest.fixture
def model_factory() -> ScanModelFactory:
    def factory(z_offset: float, temperature_compensation: TemperatureCompensationModel | None) -> ScanModel:
        poly = Polynomial([0, 1])
        poly = cast("Polynomial", poly.convert(domain=[1 / 5.5, 10]))
        config = ScanModelConfiguration(
            "test",
            poly.coef,
            poly.domain,
            z_offset=z_offset,
            reference_temperature=40,
        )

        model = ScanModel(config, temperature_compensation)
        return model

    return factory


def test_fit() -> None:
    samples = [
        Sample(raw_count=i, time=i, frequency=1 / i, position=Position(0, 0, 0), temperature=0) for i in range(1, 20)
    ]

    fit = ScanModel.fit("test", samples, 0)

    assert fit.domain[0] == 1
    assert fit.domain[1] == 19


def test_frequency_to_distance(model_factory: ScanModelFactory) -> None:
    model = model_factory(0.0, None)
    expected = 3.0

    result = model.frequency_to_distance(expected, temperature=40)

    assert result == approx(1 / expected)


def test_distance_to_frequency(model_factory: ScanModelFactory) -> None:
    model = model_factory(0.0, None)
    expected = 2.5

    result = model.distance_to_frequency(expected, temperature=40)

    assert result == approx(1 / expected)


def test_distance_to_frequency_out_of_range(model_factory: ScanModelFactory) -> None:
    model = model_factory(0, None)
    with pytest.raises(RuntimeError, match="Attempted to map out-of-range distance"):
        _ = model.distance_to_frequency(11, temperature=40)  # Out of z_range


def test_frequency_to_distance_applies_offset(model_factory: ScanModelFactory) -> None:
    model = model_factory(-0.5, None)
    frequency = 1 / 3.0

    result = model.frequency_to_distance(frequency, temperature=40)

    assert result == 2.5


def test_distance_to_frequency_applies_offset(model_factory: ScanModelFactory) -> None:
    model = model_factory(-0.5, None)
    distance = 2.5

    result = model.distance_to_frequency(distance, temperature=40)

    assert result == approx(1 / 3)


def test_frequency_to_distance_out_of_range(model_factory: ScanModelFactory) -> None:
    model = model_factory(0, None)
    low_frequency_dist = model.frequency_to_distance(1 / 500, temperature=40)  # Out of z_range
    high_frequency_dist = model.frequency_to_distance(1000000, temperature=40)  # Out of z_range

    assert low_frequency_dist == float("inf")
    assert high_frequency_dist == float("-inf")


def test_compensated_frequency_to_distance(mocker: MockerFixture, model_factory: ScanModelFactory) -> None:
    expected = 5
    temp_compensation = mocker.Mock(spec=TemperatureCompensationModel)
    temp_compensation.compensate.return_value = 1 / expected
    model = model_factory(0.0, temp_compensation)

    result = model.frequency_to_distance(42, temperature=40)

    assert result == approx(expected)


def test_distance_to_compensated_frequency(mocker: MockerFixture, model_factory: ScanModelFactory) -> None:
    expected = 5
    temp_compensation = mocker.Mock(spec=TemperatureCompensationModel)
    temp_compensation.compensate.return_value = expected
    model = model_factory(0.0, temp_compensation)

    result = model.distance_to_frequency(10, temperature=40)

    assert result == approx(expected)
