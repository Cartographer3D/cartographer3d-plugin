from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from cartographer.interfaces.printer import Position
from cartographer.macros.scan_calibrate import ScanCalibrateMacro
from tests.mocks.params import MockParams

if TYPE_CHECKING:
    from unittest.mock import Mock

    from pytest_mock import MockerFixture

    from cartographer.interfaces.configuration import Configuration
    from cartographer.interfaces.printer import Toolhead
    from cartographer.probe.probe import Probe


def test_touch_calibration_preserves_command(
    mocker: MockerFixture, probe: Probe, toolhead: Toolhead, config: Configuration
) -> None:
    params = MockParams()
    params.params["METHOD"] = "touch"
    probe.touch.perform_probe = mocker.Mock(return_value=0.5)
    macro = ScanCalibrateMacro(probe, toolhead, config)
    calibrate = mocker.patch.object(macro, "_calibrate")

    macro.run(params)

    probe.touch.perform_probe.assert_called_once_with(gcmd=params)
    assert probe.touch.perform_probe.call_args.kwargs["gcmd"] is params
    calibrate.assert_called_once_with("default")


@pytest.mark.parametrize("position", [Position(1, 2, 3), None])
def test_manual_calibration_forwards_initiating_command_and_accepts_completion(
    mocker: MockerFixture, probe: Probe, toolhead: Toolhead, config: Configuration, position: Position | None
) -> None:
    params = MockParams()
    params.params["MODEL"] = "Custom"
    completing = MockParams()
    manual_probe = cast("Mock", toolhead.manual_probe)
    clear_homing = cast("Mock", toolhead.clear_z_homing_state)
    set_z = cast("Mock", toolhead.set_z_position)
    macro = ScanCalibrateMacro(probe, toolhead, config)
    calibrate = mocker.patch.object(macro, "_calibrate")

    macro.run(params)

    manual_probe.assert_called_once()
    assert manual_probe.call_args.kwargs["gcmd"] is params
    calibrate.assert_not_called()
    set_z.assert_called_once_with(z=90)
    manual_probe.call_args.args[0](position, completing)
    if position is None:
        clear_homing.assert_called_once_with()
        calibrate.assert_not_called()
    else:
        clear_homing.assert_not_called()
        assert set_z.call_args.args == (0.1,)
        calibrate.assert_called_once_with("custom")
