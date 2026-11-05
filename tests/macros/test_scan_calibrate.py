from __future__ import annotations

from typing import TYPE_CHECKING

from cartographer.macros.scan_calibrate import ScanCalibrateMacro
from tests.mocks.params import MockParams

if TYPE_CHECKING:
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
