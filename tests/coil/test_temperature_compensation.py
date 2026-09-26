from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

from cartographer.coil.temperature_compensation import CoilTemperatureCompensationModel
from cartographer.config.fields import parse
from cartographer.interfaces.configuration import CoilCalibrationConfiguration, CoilConfiguration
from cartographer.interfaces.printer import CoilCalibrationReference

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

# Coefficients from a real calibration (Voron 2.4, 42-70 C)
COEFFS = (3.0991369076649886e-05, -13.04871896005751, -0.003917096657189427, 1544.3203560140907)
REF = CoilCalibrationReference(min_frequency=31600800 * 21250000 / 2**28, min_frequency_temperature=0)


class _Mcu:
    def get_coil_reference(self) -> CoilCalibrationReference:
        return REF


def _model(temperature_range: tuple[float, float] | None) -> CoilTemperatureCompensationModel:
    return CoilTemperatureCompensationModel(CoilCalibrationConfiguration(*COEFFS, temperature_range), _Mcu())


@pytest.mark.parametrize("frequency", [3.0e6, 3.1e6, 3.2e6])
@pytest.mark.parametrize(("source", "target"), [(45.0, 60.0), (69.0, 43.0), (50.0, 50.0)])
def test_inside_the_range_matches_the_quadratic_model(frequency: float, source: float, target: float) -> None:
    quadratic = _model(None).compensate(frequency, source, target)
    with_range = _model((42.0, 70.0)).compensate(frequency, source, target)
    assert math.isclose(with_range, quadratic, abs_tol=1e-6)


def test_outside_the_range_continues_along_the_edge_slope() -> None:
    frequency, source = 3.1e6, 50.0
    model = _model((42.0, 70.0))
    at_edge = model.compensate(frequency, source, 42.0)
    one_below = model.compensate(frequency, source, 41.0)
    ten_below = model.compensate(frequency, source, 32.0)
    # linear beyond the edge: ten degrees out is exactly ten times one degree out
    assert math.isclose(ten_below - at_edge, 10 * (one_below - at_edge), rel_tol=1e-9)
    # and it grows less than the quadratic, which keeps steepening
    quadratic_ten_below = _model(None).compensate(frequency, source, 32.0)
    assert abs(ten_below - at_edge) < abs(quadratic_ten_below - at_edge)


def _parse_calibration(mocker: MockerFixture, values: list[float]) -> CoilCalibrationConfiguration | None:
    config = mocker.Mock()
    config.error = ValueError
    config.getfloatlist = mocker.Mock(return_value=values)
    return parse(CoilConfiguration, config, name="coil", min_temp=0, max_temp=105).calibration


def test_parse_accepts_model_only_and_model_with_range(mocker: MockerFixture) -> None:
    assert _parse_calibration(mocker, list(COEFFS)) == CoilCalibrationConfiguration(*COEFFS)
    parsed = _parse_calibration(mocker, [*COEFFS, 43.6, 70.0])
    assert parsed is not None
    assert parsed.temperature_range == (43.6, 70.0)
    with pytest.raises(ValueError, match="4 or 6 values"):
        _ = _parse_calibration(mocker, [1.0, 2.0, 3.0])


def test_config_values_round_trip() -> None:
    assert CoilCalibrationConfiguration(*COEFFS).as_config_values() == list(COEFFS)
    assert CoilCalibrationConfiguration(*COEFFS, (43.6, 70.0)).as_config_values() == [*COEFFS, 43.6, 70.0]
