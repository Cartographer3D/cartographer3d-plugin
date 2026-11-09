from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock, call

from cartographer.interfaces.printer import GCodeDispatch, Mcu, Toolhead
from cartographer.macros.temperature_calibrate import TemperatureCalibrateMacro
from tests.mocks.params import MockParams

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def test_all_calibration_scripts_receive_original_command(mocker: MockerFixture) -> None:
    params = MockParams()
    mcu, toolhead, gcode = Mock(spec=Mcu), Mock(spec=Toolhead), Mock(spec=GCodeDispatch)
    toolhead.get_axis_limits.return_value = (0, 100)
    config, executor = Mock(), Mock()
    config.bed_mesh.zero_reference_position = (50, 50)
    config.general.travel_speed = 100
    macro = TemperatureCalibrateMacro(mcu, toolhead, config, gcode, executor, Mock())
    wait = mocker.patch.object(macro, "_wait_for_temperature")
    _ = mocker.patch("cartographer.macros.temperature_calibrate.scipy_helpers.raise_if_curve_fit_unavailable")
    _ = mocker.patch("cartographer.macros.temperature_calibrate.generate_filepath", return_value="samples.csv")
    _ = mocker.patch("cartographer.macros.temperature_calibrate.write_samples_to_csv")

    macro.run(params)

    assert gcode.run_gcode.call_args_list == [
        script_call
        for _ in range(3)
        for script_call in (
            call("M140 S0\nM106 S255", gcmd=params),
            call("M140 S90\nM106 S0", gcmd=params),
        )
    ] + [call("M140 S0", gcmd=params)]
    assert all(script_call.kwargs["gcmd"] is params for script_call in gcode.run_gcode.call_args_list)
    assert wait.call_args_list == [
        wait_call
        for _ in range(3)
        for wait_call in (
            call(target_temp=40, cooling=True),
            call(target_temp=39, cooling=False),
            call(target_temp=60, cooling=False),
        )
    ]
    assert mcu.register_callback.call_count == mcu.unregister_callback.call_count == 3
    executor.run.assert_called_once_with(mocker.ANY, {1: [], 2: [], 3: []}, mcu.get_coil_reference.return_value)
    config.save_coil_model.assert_called_once_with(executor.run.return_value)
