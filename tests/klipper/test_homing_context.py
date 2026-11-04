from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import Mock

import pytest

from cartographer.adapters.klipper.endstop import KlipperEndstop
from cartographer.adapters.klipper_like.gcode_context import gcode_command_context
from cartographer.adapters.klipper_like.integrator import KlipperLikeIntegrator

if TYPE_CHECKING:
    from gcode import GCodeCommand
    from pytest_mock import MockerFixture

    from cartographer.adapters.klipper.toolhead import KlipperToolhead
    from cartographer.adapters.klipper_like.toolhead import FakeRail


@pytest.fixture
def adapter(mocker: MockerFixture) -> KlipperToolhead:
    mocker.patch.dict("sys.modules", {"extras.homing": Mock(), "extras.manual_probe": Mock()})
    _ = mocker.patch("gcode.CommandError", ValueError)
    from cartographer.adapters.klipper.toolhead import KlipperToolhead

    printer = Mock()
    firmware_toolhead = Mock()
    firmware_toolhead.get_position.return_value = [10.0, 20.0, 5.0, 0.0]
    firmware_toolhead.get_status.return_value = {"axis_minimum": [0, 0, -5], "axis_maximum": [100, 100, 100]}
    printer.lookup_object.side_effect = {"toolhead": firmware_toolhead, "gcode": Mock()}.__getitem__
    config = Mock()
    config.get_printer.return_value = printer
    mcu = Mock()
    stepper = Mock()
    stepper.get_name.return_value = "stepper_z"
    stepper.get_mcu_position.return_value = 123
    mcu.dispatch.get_steppers.return_value = [stepper]
    return KlipperToolhead(config, mcu)


@pytest.mark.parametrize(("new_api", "scoped"), [(False, False), (False, True), (True, True)])
def test_probing_signature_variants(adapter: KlipperToolhead, new_api: bool, scoped: bool) -> None:
    backend = Mock()
    gcmd = Mock()

    def old_move(endstop: object, pos: list[float], speed: float) -> list[float]:
        backend(endstop, pos, speed)
        return [10, 20, 1.25, 0]

    def new_move(endstop: object, pos: list[float], speed: float, *, gcmd: GCodeCommand) -> list[float]:
        backend(endstop, pos, speed, gcmd=gcmd)
        return [10, 20, 1.25, 0]

    homing = Mock(probing_move=new_move if new_api else old_move)
    cast("Mock", adapter.printer).lookup_object.side_effect = {
        "homing": homing,
        "toolhead": adapter.toolhead,
    }.__getitem__
    with gcode_command_context(gcmd if scoped else None):
        assert adapter.z_probing_move(Mock(), speed=2.0) == 1.25
    backend.assert_called_once()
    assert backend.call_args.args[1:] == ([10.0, 20.0, -5, 0.0], 2.0)
    assert backend.call_args.kwargs == ({"gcmd": gcmd} if new_api else {})
    if new_api:
        assert backend.call_args.kwargs["gcmd"] is gcmd


@pytest.mark.parametrize("operation", ["probe", "home_end"])
def test_new_api_missing_command_has_no_side_effects(
    adapter: KlipperToolhead, mocker: MockerFixture, operation: str
) -> None:
    backend = Mock()

    def new_move(endstop: object, pos: list[float], speed: float, *, gcmd: GCodeCommand) -> list[float]:
        backend(endstop, pos, speed, gcmd=gcmd)
        return pos

    def new_homing(printer: object, *, gcmd: GCodeCommand) -> Mock:
        return backend(printer, gcmd=gcmd)

    homing = Mock(probing_move=new_move)
    cast("Mock", adapter.printer).lookup_object.side_effect = {
        "homing": homing,
        "toolhead": adapter.toolhead,
    }.__getitem__
    _ = mocker.patch("cartographer.adapters.klipper_like.toolhead.Homing", new_homing)
    firmware_toolhead = cast("Mock", adapter.toolhead)
    firmware_toolhead.reset_mock()
    with gcode_command_context(None), pytest.raises(RuntimeError, match="(?i)(active|live).*command"):
        if operation == "probe":
            _ = adapter.z_probing_move(Mock(), speed=2.0)
        else:
            adapter.z_home_end(Mock())
    backend.assert_not_called()
    assert firmware_toolhead.mock_calls == []
    cast("Mock", adapter.printer).send_event.assert_not_called()


@pytest.mark.parametrize(("new_api", "scoped"), [(False, False), (False, True), (True, True)])
def test_synthetic_home_success_visible_to_listeners(
    adapter: KlipperToolhead, mocker: MockerFixture, new_api: bool, scoped: bool
) -> None:
    backend = Mock()
    state = Mock(spec=["set_axes", "trigger_mcu_pos", "homing_successful"])
    state.homing_successful = False
    gcmd = Mock()

    def old_homing(printer: object) -> Mock:
        backend(printer)
        return state

    def new_homing(printer: object, *, gcmd: GCodeCommand) -> Mock:
        backend(printer, gcmd=gcmd)
        return state

    def listener(event: str, homing: Mock, rails: list[FakeRail]) -> None:
        assert event == "homing:home_rails_end"
        assert homing is state
        assert homing.homing_successful is True
        homing.set_axes.assert_called_once_with([2])
        assert homing.trigger_mcu_pos == {"stepper_z": 123}
        assert len(rails) == 1

    _ = mocker.patch("cartographer.adapters.klipper_like.toolhead.Homing", new_homing if new_api else old_homing)
    cast("Mock", adapter.printer).send_event.side_effect = listener
    with gcode_command_context(gcmd if scoped else None):
        adapter.z_home_end(Mock())
    backend.assert_called_once_with(adapter.printer, **({"gcmd": gcmd} if new_api else {}))
    if new_api:
        assert backend.call_args.kwargs["gcmd"] is gcmd
    cast("Mock", adapter.printer).send_event.assert_called_once()


@pytest.mark.parametrize("operation", ["probe", "home_end"])
@pytest.mark.parametrize("new_api", [False, True])
def test_backend_type_error_is_not_retried(
    adapter: KlipperToolhead, mocker: MockerFixture, operation: str, new_api: bool
) -> None:
    error = TypeError("inside backend")
    backend = Mock(side_effect=error)

    def old_move(endstop: object, pos: list[float], speed: float) -> list[float]:
        return backend(endstop, pos, speed)

    def new_move(endstop: object, pos: list[float], speed: float, *, gcmd: GCodeCommand) -> list[float]:
        return backend(endstop, pos, speed, gcmd=gcmd)

    def old_homing(printer: object) -> Mock:
        return backend(printer)

    def new_homing(printer: object, *, gcmd: GCodeCommand) -> Mock:
        return backend(printer, gcmd=gcmd)

    homing = Mock(probing_move=new_move if new_api else old_move)
    cast("Mock", adapter.printer).lookup_object.side_effect = {
        "homing": homing,
        "toolhead": adapter.toolhead,
    }.__getitem__
    _ = mocker.patch("cartographer.adapters.klipper_like.toolhead.Homing", new_homing if new_api else old_homing)
    with gcode_command_context(Mock()), pytest.raises(TypeError) as caught:
        if operation == "probe":
            _ = adapter.z_probing_move(Mock(), speed=2.0)
        else:
            adapter.z_home_end(Mock())
    assert caught.value is error
    backend.assert_called_once()
    cast("Mock", adapter.printer).send_event.assert_not_called()


@pytest.mark.parametrize("success", [False, True, None])
def test_home_end_only_notifies_successful_or_legacy_events(mocker: MockerFixture, success: bool | None) -> None:
    adapters = Mock()
    integrator = KlipperLikeIntegrator(adapters, Mock())
    homing = Mock(spec=["get_axes", "set_homed_position"])
    if success is not None:
        homing.homing_successful = success
    endstop = Mock()
    rail = Mock()
    rail.get_endstops.return_value = [(KlipperEndstop(adapters.mcu, endstop), "cartographer")]

    _ = mocker.patch("cartographer.adapters.klipper_like.integrator.setup_console_logger")
    integrator.setup()
    handler = adapters.printer.register_event_handler.call_args.args[1]
    handler(homing, [rail])

    if success is False:
        rail.get_endstops.assert_not_called()
        endstop.on_home_end.assert_not_called()
        homing.set_homed_position.assert_not_called()
    else:
        endstop.on_home_end.assert_called_once()
