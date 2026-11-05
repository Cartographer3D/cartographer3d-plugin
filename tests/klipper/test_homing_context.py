from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, cast
from unittest.mock import Mock

import pytest

from cartographer.adapters.klipper.endstop import KlipperEndstop
from cartographer.adapters.klipper_like.gcode_context import gcode_command_context, get_gcode_command
from cartographer.adapters.klipper_like.integrator import KlipperLikeIntegrator
from cartographer.interfaces.printer import Position

if TYPE_CHECKING:
    from collections.abc import Callable

    from gcode import GCodeCommand
    from pytest_mock import MockerFixture

    from cartographer.adapters.klipper.toolhead import KlipperToolhead
    from cartographer.adapters.klipper_like.toolhead import FakeRail
    from cartographer.interfaces.printer import MacroParams


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


@pytest.mark.parametrize("new_api", [False, True])
@pytest.mark.parametrize("scoped", [False, True])
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
    with gcode_command_context(Mock() if scoped else None):
        assert adapter.z_probing_move(Mock(), speed=2.0, gcmd=gcmd) == 1.25
    backend.assert_called_once()
    assert backend.call_args.args[1:] == ([10.0, 20.0, -5, 0.0], 2.0)
    assert backend.call_args.kwargs == ({"gcmd": gcmd} if new_api else {})
    if new_api:
        assert backend.call_args.kwargs["gcmd"] is gcmd


def test_kalico_acceleration_uses_direct_api(adapter: KlipperToolhead, params: MacroParams) -> None:
    from cartographer.adapters.kalico.toolhead import KalicoToolhead

    config = Mock()
    config.get_printer.return_value = adapter.printer
    kalico = KalicoToolhead(config, Mock())

    kalico.set_max_accel(100, gcmd=params)

    cast("Mock", adapter.toolhead).set_accel.assert_called_once_with(100)
    parameter = inspect.signature(kalico.set_max_accel).parameters["gcmd"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty


@pytest.mark.parametrize("operation", ["probe", "home_end"])
def test_physical_command_is_required(adapter: KlipperToolhead, operation: str) -> None:
    method = adapter.z_probing_move if operation == "probe" else adapter.z_home_end
    signature = inspect.signature(method)
    assert signature.parameters["gcmd"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["gcmd"].default is inspect.Parameter.empty
    with pytest.raises(TypeError, match="gcmd"):
        _ = signature.bind(Mock(), **({"speed": 2.0} if operation == "probe" else {}))
    cast("Mock", adapter.printer).send_event.assert_not_called()


@pytest.mark.parametrize("new_api", [False, True])
@pytest.mark.parametrize("scoped", [False, True])
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
    with gcode_command_context(Mock() if scoped else None):
        adapter.z_home_end(Mock(), gcmd=gcmd)
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
    gcmd = Mock()
    with pytest.raises(TypeError) as caught:
        if operation == "probe":
            _ = adapter.z_probing_move(Mock(), speed=2.0, gcmd=gcmd)
        else:
            adapter.z_home_end(Mock(), gcmd=gcmd)
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


@pytest.mark.parametrize("pos", [[1.0, 2.0, 3.0, 4.0], None])
@pytest.mark.parametrize("callback_error", [False, True])
def test_manual_probe_live_completion_context(
    adapter: KlipperToolhead, mocker: MockerFixture, pos: list[float] | None, callback_error: bool
) -> None:
    initiating = Mock(expired=False)
    completing = Mock()
    previous = Mock()
    backend = Mock()
    callbacks: list[Callable[[list[float] | None, GCodeCommand], None]] = []

    def new_helper(
        printer: object,
        gcmd: GCodeCommand,
        finalize_callback: Callable[[list[float] | None, GCodeCommand], None],
        pass_gcmd: bool = False,
    ) -> None:
        assert gcmd is initiating
        assert not initiating.expired
        assert pass_gcmd is True
        backend(printer, gcmd)
        callbacks.append(finalize_callback)

    error = ValueError("completion failed")
    seen: list[Position | None] = []

    def finalize(position: Position | None) -> None:
        assert initiating.expired
        assert get_gcode_command() is completing
        seen.append(position)
        if callback_error:
            raise error

    _ = mocker.patch("cartographer.adapters.klipper_like.toolhead.ManualProbeHelper", new_helper)
    with gcode_command_context(initiating):
        adapter.manual_probe(finalize)
        assert get_gcode_command() is initiating
    initiating.expired = True
    with gcode_command_context(previous):
        if callback_error:
            with pytest.raises(ValueError) as caught:
                callbacks[0](pos, completing)
            assert caught.value is error
        else:
            callbacks[0](pos, completing)
        assert get_gcode_command() is previous
    assert seen == [Position(1.0, 2.0, 3.0) if pos else None]
    backend.assert_called_once_with(adapter.printer, initiating)
    cast("Mock", adapter.printer).lookup_object.assert_not_called()


@pytest.mark.parametrize("pos", [[1.0, 2.0, 3.0, 4.0], None])
@pytest.mark.parametrize("callback_error", [False, True])
def test_manual_probe_legacy_completion_clears_context(
    adapter: KlipperToolhead, mocker: MockerFixture, pos: list[float] | None, callback_error: bool
) -> None:
    initiating = Mock(expired=False)
    previous = Mock()
    backend = Mock()
    callbacks: list[Callable[[list[float] | None], None]] = []
    gcode = cast("Mock", adapter.printer.lookup_object("gcode"))

    def old_helper(
        printer: object, gcmd: GCodeCommand, finalize_callback: Callable[[list[float] | None], None]
    ) -> None:
        assert gcmd is gcode.create_gcode_command.return_value
        backend(printer, gcmd)
        callbacks.append(finalize_callback)

    error = ValueError("completion failed")
    seen: list[Position | None] = []

    def finalize(position: Position | None) -> None:
        assert initiating.expired
        assert get_gcode_command() is None
        seen.append(position)
        if callback_error:
            raise error

    _ = mocker.patch("cartographer.adapters.klipper_like.toolhead.ManualProbeHelper", old_helper)
    with gcode_command_context(initiating):
        adapter.manual_probe(finalize)
        assert get_gcode_command() is initiating
    initiating.expired = True
    with gcode_command_context(previous):
        if callback_error:
            with pytest.raises(ValueError) as caught:
                callbacks[0](pos)
            assert caught.value is error
        else:
            callbacks[0](pos)
        assert get_gcode_command() is previous
    assert seen == [Position(1.0, 2.0, 3.0) if pos else None]
    gcode.create_gcode_command.assert_called_once_with("", "", {})
    backend.assert_called_once_with(adapter.printer, gcode.create_gcode_command.return_value)


def test_manual_probe_new_helper_requires_live_context(adapter: KlipperToolhead, mocker: MockerFixture) -> None:
    backend = Mock()
    finalize = Mock()

    def new_helper(printer: object, gcmd: GCodeCommand, callback: object, pass_gcmd: bool = False) -> None:
        backend(printer, gcmd, callback, pass_gcmd)

    _ = mocker.patch("cartographer.adapters.klipper_like.toolhead.ManualProbeHelper", new_helper)
    with gcode_command_context(None), pytest.raises(RuntimeError, match="(?i)(active|live).*command"):
        adapter.manual_probe(finalize)
    backend.assert_not_called()
    finalize.assert_not_called()
    cast("Mock", adapter.printer).lookup_object.assert_not_called()


@pytest.mark.parametrize("new_api", [False, True])
def test_manual_probe_backend_type_error_is_not_retried(
    adapter: KlipperToolhead, mocker: MockerFixture, new_api: bool
) -> None:
    error = TypeError("inside manual probe backend")
    backend = Mock(side_effect=error)
    initiating = Mock()

    def old_helper(printer: object, gcmd: GCodeCommand, callback: object) -> None:
        backend(printer, gcmd, callback)

    def new_helper(printer: object, gcmd: GCodeCommand, callback: object, pass_gcmd: bool = False) -> None:
        assert gcmd is initiating
        assert pass_gcmd is True
        backend(printer, gcmd, callback, pass_gcmd)

    _ = mocker.patch(
        "cartographer.adapters.klipper_like.toolhead.ManualProbeHelper", new_helper if new_api else old_helper
    )
    with gcode_command_context(initiating), pytest.raises(TypeError) as caught:
        adapter.manual_probe(Mock())
    assert caught.value is error
    backend.assert_called_once()
