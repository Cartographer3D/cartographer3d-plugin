from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING, Callable, cast, final
from unittest.mock import Mock

import pytest

from cartographer.adapters.kalico.integrator import KalicoIntegrator
from cartographer.adapters.kalico.probe import KalicoCartographerProbe
from cartographer.adapters.klipper.endstop import KlipperEndstop
from cartographer.core import MacroRegistration
from cartographer.extra import load_config
from cartographer.interfaces.printer import Position
from cartographer.macros.bed_mesh.scan_mesh import BedMeshCalibrateMacro
from tests.mocks.params import MockParams

if TYPE_CHECKING:
    from gcode import GCodeCommand
    from pytest_mock import MockerFixture


@final
class Command(MockParams):
    error = ValueError

    def __init__(self, command: str, commandline: str, params: dict[str, str]) -> None:
        super().__init__()
        self.command = command
        self.commandline = commandline
        self.params = params
        self.respond_info = Mock()

    def get_command(self) -> str:
        return self.command

    def get_commandline(self) -> str:
        return self.commandline

    def get_command_parameters(self) -> dict[str, str]:
        return self.params


PROBE_COMMANDS = ("PROBE", "PROBE_ACCURACY", "QUERY_PROBE", "Z_OFFSET_APPLY_PROBE")


@pytest.fixture
def host(mocker: MockerFixture) -> Mock:
    host = Mock()
    host.objects = {"gcode": host.gcode}

    def lookup(name: str, default: object = None) -> object:
        return host.objects.get(name, default)

    host.printer.lookup_object.side_effect = lookup
    host.printer.command_error = ValueError

    def config_error(message: str) -> ValueError:
        return ValueError(message)

    host.config.wrapper.error.side_effect = config_error
    host.config.wrapper.has_section.return_value = False
    host.gcode.create_gcode_command.side_effect = Command
    host.gcode.register_command.return_value = None
    host.device.config.general.register_as_probe = True
    host.device.config.general.lift_speed = 6.0
    host.device.macros = []
    host.device.probe_macros = [MacroRegistration(name, Mock(spec=["run", "description"])) for name in PROBE_COMMANDS]
    module = ModuleType("extras.probe")
    host.api.get_list.return_value = host.registry
    module.__dict__["ProbeList"] = host.api
    host.module = module
    _ = mocker.patch("cartographer.adapters.kalico.integrator.import_module", return_value=module)
    return host


def register(host: Mock, is_default: bool = True) -> tuple[KalicoIntegrator, KalicoCartographerProbe]:
    host.device.config.general.register_as_probe = is_default
    integrator = KalicoIntegrator(host)
    integrator.register_probe(host.device)
    probe = host.registry.add_probe_object.call_args.args[0]
    assert type(probe) is KalicoCartographerProbe
    return integrator, probe


def command_handler(host: Mock) -> Callable[[Command], None]:
    return host.gcode.register_command.call_args.args[1]


def assert_copy(host: Mock, original: Command, received: Command) -> None:
    assert received is not original
    assert received.params is not original.params
    assert received.params == {key: value for key, value in original.params.items() if key != "PROBE"}
    assert received.command == original.command
    assert received.commandline == original.commandline
    host.gcode.create_gcode_command.assert_called_once_with(original.command, original.commandline, received.params)


@pytest.mark.parametrize("is_default", [True, False])
def test_config_registers_real_probe(mocker: MockerFixture, host: Mock, is_default: bool) -> None:
    host.device.config.general.register_as_probe = is_default
    integrator = KalicoIntegrator(host)
    _ = mocker.patch("cartographer.extra.init_runtime", return_value=(host, integrator))
    _ = mocker.patch("cartographer.extra.PrinterCartographer", return_value=host.device)
    for method in ("setup", "register_coil_temperature_sensor", "register_endstop_pin"):
        _ = mocker.patch.object(integrator, method)
    assert load_config(host.config.wrapper) is host.device
    probe = host.registry.add_probe_object.call_args.args[0]
    assert type(probe) is KalicoCartographerProbe
    assert probe.name == probe.probe_name == "cartographer"
    assert probe.printer is host.printer
    assert probe.is_default_probe is is_default
    assert probe.probe is host.device.scan_mode
    assert type(probe.mcu_probe) is KlipperEndstop
    assert probe.mcu_probe.mcu is host.mcu
    assert probe.mcu_probe.endstop is host.device.scan_mode
    host.api.get_list.assert_called_once_with(host.printer)
    host.registry.add_probe_object.assert_called_once_with(probe, host.config.wrapper)
    host.printer.add_object.assert_not_called()


@pytest.mark.parametrize(
    ("missing", "is_default"),
    [("module", True), ("module", False), ("class", True), ("class", False), ("api", True), ("api", False)],
)
def test_legacy_fallback(mocker: MockerFixture, host: Mock, missing: str, is_default: bool) -> None:
    if missing == "module":
        _ = mocker.patch(
            "cartographer.adapters.kalico.integrator.import_module",
            side_effect=ModuleNotFoundError(name="extras.probe"),
        )
    elif missing == "class":
        del host.module.ProbeList
    else:
        host.api.get_command_probe = None
    host.device.config.general.register_as_probe = is_default
    KalicoIntegrator(host).register_probe(host.device)
    if is_default:
        name, probe = host.printer.add_object.call_args.args
        assert name == "probe" and type(probe) is KalicoCartographerProbe
        assert not hasattr(probe, "mcu_probe")
        host.printer.add_object.assert_called_once()
    else:
        host.printer.add_object.assert_not_called()
    host.registry.add_probe_object.assert_not_called()


def test_dependency_and_registry_errors_propagate(mocker: MockerFixture, host: Mock) -> None:
    error = ModuleNotFoundError(name="dependency")
    importer = mocker.patch("cartographer.adapters.kalico.integrator.import_module", side_effect=error)
    with pytest.raises(ModuleNotFoundError) as caught:
        _ = KalicoIntegrator(host)
    assert caught.value is error
    importer.side_effect = None
    importer.return_value = host.module
    conflict = ValueError("native registry conflict")
    host.registry.add_probe_object.side_effect = conflict
    with pytest.raises(ValueError) as registry_error:
        _ = register(host)
    assert registry_error.value is conflict
    host.printer.add_object.assert_not_called()


@pytest.mark.parametrize("is_default", [True, False])
def test_four_mux_commands_copy_only_selector(host: Mock, is_default: bool) -> None:
    integrator, _ = register(host, is_default)
    if is_default:
        for registration in host.device.probe_macros:
            integrator.register_macro(registration)
    calls = host.gcode.register_mux_command.call_args_list
    assert [(c.args[0], c.args[1], c.args[2]) for c in calls] == [
        (name, "PROBE", key)
        for name in PROBE_COMMANDS
        for key in (["cartographer", None] if is_default else ["cartographer"])
    ]
    for registration in host.device.probe_macros:
        call = next(c for c in calls if c.args[0] == registration.name)
        assert call.kwargs["desc"] == registration.macro.description
        if is_default:
            assert next(c.args[3] for c in calls if c.args[:3] == (registration.name, "PROBE", None)) is call.args[3]
        original = Command(registration.name, "original line", {"PROBE": "cartographer", "EXTRA": "retained"})
        call.args[3](original)
        assert_copy(host, original, registration.macro.run.call_args.args[0])
        assert original.params == {"PROBE": "cartographer", "EXTRA": "retained"}
        host.gcode.create_gcode_command.reset_mock()
    host.gcode.register_command.assert_not_called()


@pytest.mark.parametrize("other_first", [True, False])
def test_named_mux_preserves_other_and_default_callbacks(host: Mock, other_first: bool) -> None:
    other = Mock()
    existing = {(name, key): other for name in PROBE_COMMANDS for key in ("other", None)}
    callbacks: dict[tuple[str, str | None], Callable[[Command], None]] = dict(existing) if other_first else {}

    def capture(name: str, key: str, value: str | None, handler: Callable[[Command], None], **_kwargs: object) -> None:
        assert key == "PROBE"
        assert (name, value) not in callbacks
        callbacks[name, value] = handler

    host.gcode.register_mux_command.side_effect = capture
    _ = register(host, False)
    if not other_first:
        assert not callbacks.keys() & existing.keys()
        callbacks.update(existing)
    for name in PROBE_COMMANDS:
        assert callbacks[name, "other"] is callbacks[name, None] is other
    host.gcode.register_command.assert_not_called()


@pytest.mark.parametrize(("legacy", "name"), [(True, "PROBE"), (False, "CARTOGRAPHER_SCAN_PROBE")])
def test_direct_commands_use_shared_route(host: Mock, legacy: bool, name: str) -> None:
    if legacy:
        host.api.get_command_probe = None
    integrator, macro = KalicoIntegrator(host), Mock(spec=["run", "description"])
    integrator.register_probe(host.device)
    integrator.register_macro(MacroRegistration(name, macro))
    original = Command(name, name, {"PROBE": "retained"})
    command_handler(host)(original)
    macro.run.assert_called_once_with(original)
    host.gcode.register_mux_command.assert_not_called()
    host.gcode.create_gcode_command.assert_not_called()


@pytest.mark.parametrize(
    ("selection", "params", "route"),
    [
        ("cartographer", {"PROBE": "cartographer"}, "copy"),
        ("cartographer", {"PROBE": "cartographer", "METHOD": "scan"}, "copy"),
        ("cartographer", {}, "copy"),
        ("cartographer", {"METHOD": "scan"}, "copy"),
        ("cartographer", {"PROBE": "cartographer", "METHOD": "automatic"}, "fallback"),
        ("other", {"PROBE": "other"}, "native"),
        ("other", {"PROBE": "other", "METHOD": "automatic"}, "native"),
        ("other", {}, "native"),
        ("other", {"METHOD": "automatic"}, "native"),
        ("none", {"METHOD": "manual"}, "native"),
        ("cartographer", {"PROBE": "cartographer", "METHOD": "manual"}, "native"),
        ("other", {"PROBE": "other", "METHOD": "scan"}, "scan error"),
        ("unknown", {"PROBE": "unknown"}, "unknown"),
        ("unknown", {"PROBE": "unknown", "METHOD": "manual"}, "unknown"),
        ("none", {}, "No default probe"),
    ],
)
def test_mesh_routes_original_or_copy(
    mocker: MockerFixture, host: Mock, selection: str, params: dict[str, str], route: str
) -> None:
    integrator, probe = register(host, selection == "cartographer" and "PROBE" not in params)
    macro, native = Mock(spec=BedMeshCalibrateMacro, adapter=host.mesh), Mock(side_effect=host.toolhead.move)
    host.gcode.register_command.return_value = native
    _ = mocker.patch("cartographer.adapters.klipper_like.integrator.GCodeCommand", Command)
    integrator.register_macro(MacroRegistration("BED_MESH_CALIBRATE", macro))
    assert host.gcode.register_command.call_args_list[0].args == ("BED_MESH_CALIBRATE", None)
    fallback = macro.set_fallback_macro.call_args.args[0]
    macro.run.side_effect = fallback.run if route == "fallback" else host.mesh.clear_mesh
    selected = {"cartographer": probe, "other": object(), "none": None, "unknown": None}[selection]
    host.registry.get_command_probe.return_value = selected
    if selection == "unknown":
        host.registry.get_command_probe.side_effect = ValueError("unknown")
    elif route == "No default probe":
        host.registry.get_command_probe.side_effect = [None, ValueError(route)]
    original = Command("BED_MESH_CALIBRATE", "original mesh line", {**params, "PROFILE": "retained"})
    before = original.params.copy()
    if route in ("scan error", "unknown", "No default probe"):
        with pytest.raises(ValueError, match="requires the Cartographer" if route == "scan error" else route):
            command_handler(host)(original)
        macro.run.assert_not_called()
        native.assert_not_called()
        host.gcode.create_gcode_command.assert_not_called()
        host.mesh.clear_mesh.assert_not_called()
        host.mesh.apply_mesh.assert_not_called()
        host.toolhead.move.assert_not_called()
    else:
        command_handler(host)(original)
        if route == "copy":
            assert_copy(host, original, macro.run.call_args.args[0])
            native.assert_not_called()
        else:
            native.assert_called_once_with(original)
            assert native.call_args.args[0] is original
            host.gcode.create_gcode_command.assert_not_called()
            if route == "fallback":
                macro.run.assert_called_once_with(original)
            else:
                macro.run.assert_not_called()
    assert host.registry.get_command_probe.call_args_list[0].args == (original, None)
    assert original.params == before


def test_missing_native_errors_but_named_scan_works(host: Mock) -> None:
    integrator, probe = register(host, False)
    host.registry.get_command_probe.return_value = probe
    macro = Mock(spec=BedMeshCalibrateMacro, adapter=host.mesh)
    macro.run.side_effect = host.mesh.clear_mesh
    integrator.register_macro(MacroRegistration("BED_MESH_CALIBRATE", macro))
    original = Command("BED_MESH_CALIBRATE", "mesh", {"PROBE": "cartographer", "METHOD": "automatic"})
    with pytest.raises(ValueError, match="native BED_MESH_CALIBRATE.*unavailable"):
        command_handler(host)(original)
    macro.run.assert_not_called()
    host.toolhead.move.assert_not_called()
    macro.adapter.clear_mesh.assert_not_called()
    original.params["METHOD"] = "scan"
    command_handler(host)(original)
    assert_copy(host, original, macro.run.call_args.args[0])
    assert original.params == {"PROBE": "cartographer", "METHOD": "scan"}
    macro.set_fallback_macro.assert_not_called()


def test_real_probe_native_rapid_diagnostic_and_point_contract(host: Mock) -> None:
    _, probe = register(host)
    assert not hasattr(probe.mcu_probe, "begin_collect_distance")
    command = Command("BED_MESH_CALIBRATE", "rapid scan", {"METHOD": "rapid_scan", "LIFT_SPEED": "8"})
    # Native consumer-shaped diagnostic: removing the real adapter's name must fail here.
    if command.get("METHOD") == "rapid_scan" and not hasattr(probe.mcu_probe, "begin_collect_distance"):
        command.respond_info(f"METHOD=rapid_scan not supported for probe {probe.name}, using automatic")
    command.respond_info.assert_called_once_with(
        "METHOD=rapid_scan not supported for probe cartographer, using automatic"
    )
    host.device.scan_mode.perform_probe.assert_not_called()
    host.toolhead.move.assert_not_called()
    host.toolhead.get_position.return_value = Position(10, 20, 5)
    host.device.scan_mode.perform_probe.return_value = 1.5
    host.device.scan_mode.offset.as_tuple.return_value = (1.0, 2.0, 3.0)
    native_command = cast("GCodeCommand", cast("object", command))
    assert probe.run_probe(native_command, object()) == [10, 20, 1.5]
    host.device.scan_mode.perform_probe.assert_called_once_with()
    assert probe.get_offsets() == (1.0, 2.0, 3.0)
    assert probe.get_lift_speed(native_command) == 8 and probe.get_lift_speed() == 6
    for triggered in (True, False):
        host.device.scan_mode.query_is_triggered.return_value = triggered
        assert probe.mcu_probe.query_endstop(12.5) == int(triggered)
    host.device.scan_mode.query_is_triggered.assert_called_with(12.5)


@pytest.mark.parametrize("selection", ["selected", "default", "other", "missing"])
def test_z_guard_connect_shared_boundary(mocker: MockerFixture, host: Mock, selection: str) -> None:
    integrator, probe = register(host)
    _ = mocker.patch.object(integrator, "_configure_macro_logger")
    integrator.setup()
    host.config.wrapper.has_section.return_value = True
    original, motion = Mock(), host.toolhead.move
    helper = Mock(load_probe=original)
    host.objects["z_calibration"] = None if selection == "missing" else helper
    callback = next(
        c.args[1] for c in host.printer.register_event_handler.call_args_list if c.args[0] == "klippy:connect"
    )
    if selection == "missing":
        with pytest.raises(ValueError, match="z_calibration.*load_probe"):
            callback()
    else:
        callback()
        selected = object() if selection == "other" else probe
        host.registry.get_command_probe.return_value = selected

        def native_command(command: Command) -> object:
            result = helper.load_probe(host.registry.get_command_probe(command))
            motion()
            return result

        params = {} if selection == "default" else {"PROBE": "other" if selection == "other" else "cartographer"}
        command = Command("RENAMED_CALIBRATE_Z", "native alias", {**params, "OPTIONAL_OVERRIDE": "7"})
        if selection == "other":
            assert native_command(command) is original.return_value
            original.assert_called_once_with(selected)
            motion.assert_called_once_with()
        else:
            with pytest.raises(ValueError, match="native Z calibration.*unsupported.*Cartographer calibration"):
                _ = native_command(command)
    if selection != "other":
        original.assert_not_called()
        motion.assert_not_called()


@pytest.mark.parametrize("legacy", [False, True])
def test_z_guard_inactive(mocker: MockerFixture, host: Mock, legacy: bool) -> None:
    if legacy:
        host.api.get_command_probe = None
    integrator = KalicoIntegrator(host)
    _ = mocker.patch.object(integrator, "_configure_macro_logger")
    integrator.setup()
    helper = Mock()
    original = helper.load_probe
    host.objects["z_calibration"] = helper
    callbacks = [c.args[1] for c in host.printer.register_event_handler.call_args_list if c.args[0] == "klippy:connect"]
    if legacy:
        assert not callbacks
    else:
        assert len(callbacks) == 1
        callbacks[0]()
    assert helper.load_probe is original


@pytest.mark.parametrize(
    ("is_default", "selector", "reject"),
    [(True, None, True), (False, None, False), (False, "cartographer", True), (True, "other", False)],
)
def test_nozzle_config(host: Mock, is_default: bool, selector: str | None, reject: bool) -> None:
    host.config.wrapper.has_section.return_value = True
    section = host.config.wrapper.getsection.return_value
    section.get.return_value = selector
    section.error.side_effect = host.config.wrapper.error.side_effect
    if reject:
        with pytest.raises(ValueError, match="nozzle_cleanup.*unsupported.*Cartographer"):
            _ = register(host, is_default)
        host.api.get_list.assert_not_called()
        host.registry.add_probe_object.assert_not_called()
        host.printer.add_object.assert_not_called()
    else:
        _ = register(host, is_default)
        host.registry.add_probe_object.assert_called_once()
    section.get.assert_called_once_with("probe", None)
    section.getfloat.assert_not_called()
