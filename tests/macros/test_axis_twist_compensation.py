from __future__ import annotations

from typing import TYPE_CHECKING

from cartographer.macros.axis_twist_compensation import AxisTwistCompensationMacro, CalibrationOptions

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

    from cartographer.interfaces.configuration import Configuration
    from cartographer.interfaces.printer import MacroParams, Toolhead
    from cartographer.macros.axis_twist_compensation import AxisTwistCompensationAdapter
    from cartographer.probe.probe import Probe


def test_calibration_probes_preserve_command(
    mocker: MockerFixture,
    probe: Probe,
    toolhead: Toolhead,
    config: Configuration,
    params: MacroParams,
    axis_twist_compensation_adapter: AxisTwistCompensationAdapter,
) -> None:
    adapter = axis_twist_compensation_adapter
    adapter.available_axes = ("x", "y")
    adapter.move_height = 5
    adapter.speed = 100
    options = mocker.patch.object(adapter, "get_calibration_options", return_value=CalibrationOptions(10, 20, 10))
    scan = mocker.patch.object(probe.scan, "perform_probe", return_value=1)
    touch = mocker.patch.object(probe.touch, "perform_probe", return_value=0.5)
    macro = AxisTwistCompensationMacro(probe, toolhead, adapter, config)

    macro.run(params)

    options.assert_called_once_with("x")
    for method in (scan, touch):
        calls = method.call_args_list
        assert len(calls) == 5
        assert all(call.kwargs["gcmd"] is params for call in calls)
