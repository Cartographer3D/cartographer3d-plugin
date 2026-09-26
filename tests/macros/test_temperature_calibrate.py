from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from cartographer.interfaces.printer import Position
from cartographer.macros.temperature_calibrate import TemperatureCalibrateMacro, collection_targets
from tests.mocks.params import MockParams

if TYPE_CHECKING:
    from unittest.mock import Mock

    from pytest_mock import MockerFixture


def test_collection_targets_step_up_to_max() -> None:
    assert collection_targets(39, 70, 0) == [70]
    assert collection_targets(39, 70, 10) == [49, 59, 69, 70]


def _run(mocker: MockerFixture, touch: Mock | None, **params: str) -> tuple[Mock, Mock]:
    toolhead = mocker.Mock()
    toolhead.get_position = mocker.Mock(return_value=Position(175, 175, 1.2))
    toolhead.get_axis_limits = mocker.Mock(return_value=(0, 300))
    config = mocker.Mock()
    config.bed_mesh.zero_reference_position = (175, 175)
    macro = TemperatureCalibrateMacro(
        mocker.Mock(), toolhead, config, mocker.Mock(), mocker.Mock(), mocker.Mock(), touch=touch
    )
    _ = mocker.patch("cartographer.macros.temperature_calibrate.scipy_helpers.raise_if_curve_fit_unavailable")
    _ = mocker.patch("cartographer.macros.temperature_calibrate.write_samples_to_csv")
    wait = mocker.patch.object(macro, "_wait_for_temperature")
    macro_params = MockParams()
    macro_params.params = {"MIN_TEMP": "40", "MAX_TEMP": "60", "BED_TEMP": "110", **params}
    macro.run(macro_params)
    return toolhead, wait


def test_touch_step_rereferences_z_before_each_step(mocker: MockerFixture) -> None:
    touch = mocker.Mock()
    touch.perform_probe = mocker.Mock(return_value=0.03)

    toolhead, wait = _run(mocker, touch, TOUCH_STEP="10")

    heating = [c.kwargs["target_temp"] for c in wait.call_args_list if c.kwargs["cooling"] is False]
    # per phase: reach 39, then 49 / 59 / 60 with a touch before each; 3 phases
    assert heating == [39, 49, 59, 60] * 3
    assert touch.perform_probe.call_count == 9
    assert toolhead.set_z_position.call_args_list == [mocker.call(1.2 - 0.03)] * 9


def test_without_touch_step_never_touches(mocker: MockerFixture) -> None:
    touch = mocker.Mock()

    toolhead, wait = _run(mocker, touch)

    heating = [c.kwargs["target_temp"] for c in wait.call_args_list if c.kwargs["cooling"] is False]
    assert heating == [39, 60] * 3
    touch.perform_probe.assert_not_called()
    toolhead.set_z_position.assert_not_called()


def test_touch_step_without_touch_probe_is_rejected(mocker: MockerFixture) -> None:
    with pytest.raises(RuntimeError, match="TOUCH_STEP"):
        _ = _run(mocker, None, TOUCH_STEP="5")
