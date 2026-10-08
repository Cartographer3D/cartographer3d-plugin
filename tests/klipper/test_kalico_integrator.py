from __future__ import annotations

from dataclasses import replace
from types import ModuleType
from typing import TYPE_CHECKING, Callable, cast, final
from unittest.mock import Mock

import pytest

from cartographer.adapters.kalico.integrator import KalicoIntegrator
from cartographer.adapters.kalico.probe import KalicoCartographerProbe
from cartographer.adapters.klipper.endstop import KlipperEndstop
from cartographer.core import MacroRegistration, PrinterCartographer
from cartographer.extra import load_config
from cartographer.interfaces.configuration import Configuration, ScanModelConfiguration
from cartographer.interfaces.printer import Position
from cartographer.macros.bed_mesh.scan_mesh import BedMeshCalibrateMacro
from tests.mocks.params import MockParams

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


@final
class GCodeCommand(MockParams):
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


class GCodeDispatch:
    """Mux registration and case-sensitive dispatch, without firmware imports."""

    def __init__(self) -> None:
        self.commands: dict[str, Callable[[GCodeCommand], None]] = {}
        self.mux_commands: dict[str, tuple[str, dict[str | None, Callable[[GCodeCommand], None]]]] = {}
        self.clones: list[GCodeCommand] = []
        self.descriptions: dict[str, str | None] = {}
        self.removals: list[str] = []

    def register_command(
        self, name: str, handler: Callable[[GCodeCommand], None] | None, desc: str | None = None
    ) -> Callable[[GCodeCommand], None] | None:
        if handler is None:
            self.removals.append(name)
            return self.commands.pop(name, None)
        if name in self.commands:
            message = "Command already registered"
            raise ValueError(message)
        self.commands[name] = handler
        self.descriptions[name] = desc
        return None

    def register_mux_command(
        self,
        name: str,
        key: str,
        value: str | None,
        handler: Callable[[GCodeCommand], None],
        desc: str | None = None,
    ) -> None:
        if name not in self.mux_commands:

            def dispatch(gcmd: GCodeCommand) -> None:
                mux_key, values = self.mux_commands[name]
                if None not in values and mux_key not in gcmd.params:
                    message = f"Missing {mux_key}"
                    raise gcmd.error(message)
                selector = gcmd.get(mux_key, None)
                if selector not in values:
                    message = f"Invalid {mux_key}: {selector}"
                    raise gcmd.error(message)
                values[selector](gcmd)

            _ = self.register_command(name, dispatch, desc)
            self.mux_commands[name] = (key, {})
        old_key, values = self.mux_commands[name]
        if key != old_key or value in values:
            message = "Conflicting mux registration"
            raise ValueError(message)
        values[value] = handler

    def create_gcode_command(self, command: str, commandline: str, params: dict[str, str]) -> GCodeCommand:
        clone = GCodeCommand(command, commandline, params)
        self.clones.append(clone)
        return clone

    def dispatch(self, name: str, params: dict[str, str]) -> GCodeCommand:
        gcmd = GCodeCommand(name, f"{name} original commandline", params)
        self.commands[name](gcmd)
        return gcmd


_MISSING_DEFAULT = object()


class ProbeList:
    """Registration-shaped registry; the registry alone owns the default alias."""

    def __init__(self) -> None:
        self.probes: dict[str, object] = {}
        self.default_probe: object | None = None
        self.configs: list[Mock] = []
        self.selections: list[tuple[GCodeCommand, object]] = []

    @staticmethod
    def get_list(printer: Mock) -> ProbeList:
        registry = printer.lookup_object("probe_list", None)
        if registry is None:
            registry = ProbeList()
            printer.add_object("probe_list", registry)
        return registry

    def add_probe_object(self, obj: KalicoCartographerProbe, config: Mock) -> KalicoCartographerProbe:
        if obj.probe_name in self.probes:
            message = f"Duplicate probe name {obj.probe_name}"
            raise config.error(message)
        if obj.is_default_probe:
            if config.get_name() != "probe" and config.has_section("probe"):
                message = "Conflicting [probe] section"
                raise config.error(message)
            assert obj.printer is not None
            obj.printer.add_object("probe", obj)
            self.default_probe = obj
        self.probes[obj.probe_name] = obj
        self.configs.append(config)
        return obj

    def get_all(self) -> dict[str, object]:
        return self.probes

    def get_default_probe(self) -> object | None:
        return self.default_probe

    def get_command_probe(
        self,
        gcmd: GCodeCommand,
        default: object = _MISSING_DEFAULT,
        no_default_error: str = "No default probe registered, explicitly select probe by passing `PROBE=`",
    ) -> object | None:
        self.selections.append((gcmd, default))
        name = gcmd.get("PROBE", None)
        if name is not None:
            if name not in self.probes:
                message = f"Unknown requested probe {name}"
                raise gcmd.error(message)
            return self.probes[name]
        if self.default_probe is not None:
            return self.default_probe
        if default is _MISSING_DEFAULT:
            raise gcmd.error(no_default_error)
        return default


def test_registry_no_default_errors() -> None:
    registry = ProbeList()
    command = GCodeCommand("PROBE", "PROBE", {})

    with pytest.raises(ValueError) as error:
        _ = registry.get_command_probe(command)
    assert str(error.value) == "No default probe registered, explicitly select probe by passing `PROBE=`"

    with pytest.raises(ValueError) as error:
        _ = registry.get_command_probe(command, no_default_error="Select a mesh probe")
    assert str(error.value) == "Select a mesh probe"
    assert registry.get_command_probe(command, None, no_default_error="Select a mesh probe") is None


@pytest.fixture
def adapters() -> Mock:
    host = Mock()
    objects: dict[str, object] = {"gcode": GCodeDispatch()}

    def add_object(name: str, obj: object) -> None:
        if name in objects:
            message = f"Duplicate printer object {name}"
            raise ValueError(message)
        objects[name] = obj

    def lookup_object(name: str, default: object = None) -> object:
        return objects.get(name, default)

    host.printer.lookup_object.side_effect = lookup_object
    host.printer.add_object.side_effect = add_object
    host.config.wrapper.get_name.return_value = "cartographer"
    host.config.wrapper.has_section.return_value = False

    def config_error(message: str) -> ValueError:
        return ValueError(message)

    host.config.wrapper.error.side_effect = config_error
    return host


@pytest.fixture
def cartographer() -> Mock:
    device = Mock()
    device.config.general.register_as_probe = True
    device.macros = []
    device.probe_macros = []
    return device


@pytest.fixture
def registry_module(mocker: MockerFixture) -> ModuleType:
    module = ModuleType("extras.probe")
    module.__dict__["ProbeList"] = ProbeList
    _ = mocker.patch("cartographer.adapters.kalico.integrator.import_module", return_value=module)
    return module


@pytest.mark.parametrize("is_default", [True, False])
def test_registers_actual_probe_during_config_load(
    mocker: MockerFixture, adapters: Mock, cartographer: Mock, registry_module: ModuleType, is_default: bool
) -> None:
    del registry_module
    cartographer.config.general.register_as_probe = is_default
    integrator = KalicoIntegrator(adapters)
    adapters.printer.add_object.assert_not_called()
    assert adapters.printer.lookup_object("probe_list", None) is None
    _ = mocker.patch("cartographer.extra.init_runtime", return_value=(adapters, integrator))
    _ = mocker.patch("cartographer.extra.PrinterCartographer", return_value=cartographer)
    _ = mocker.patch.object(integrator, "setup")
    _ = mocker.patch.object(integrator, "register_coil_temperature_sensor")
    _ = mocker.patch.object(integrator, "register_endstop_pin")

    assert load_config(adapters.config.wrapper) is cartographer

    registry = adapters.printer.lookup_object("probe_list", None)
    probe = registry.get_all()["cartographer"]
    assert type(probe) is KalicoCartographerProbe
    assert probe.probe_name == "cartographer"
    assert probe.is_default_probe is is_default
    assert probe.printer is adapters.printer
    assert probe.probe is cartographer.scan_mode
    assert registry.configs == [adapters.config.wrapper]
    assert registry.get_default_probe() is (probe if is_default else None)
    assert adapters.printer.lookup_object("probe", None) is (probe if is_default else None)
    assert [call.args[0] for call in adapters.printer.add_object.call_args_list] == (
        ["probe_list", "probe"] if is_default else ["probe_list"]
    )


@pytest.mark.parametrize("is_default", [True, False])
@pytest.mark.parametrize("missing", ["module", "class", "get_list", "add_probe_object", "get_command_probe"])
def test_missing_registry_uses_legacy_registration(
    mocker: MockerFixture, adapters: Mock, cartographer: Mock, is_default: bool, missing: str
) -> None:
    module = ModuleType("extras.probe")
    if missing in ("get_list", "add_probe_object", "get_command_probe"):
        module.__dict__["ProbeList"] = type("IncompleteProbeList", (ProbeList,), {missing: None})
    importer = mocker.patch("cartographer.adapters.kalico.integrator.import_module", return_value=module)
    if missing == "module":
        importer.side_effect = ModuleNotFoundError("No probe module", name="extras.probe")
    cartographer.config.general.register_as_probe = is_default

    KalicoIntegrator(adapters).register_probe(cartographer)

    if is_default:
        adapters.printer.add_object.assert_called_once()
        name, probe = adapters.printer.add_object.call_args.args
        assert name == "probe"
        assert type(probe) is KalicoCartographerProbe
        assert not hasattr(probe, "mcu_probe")
    else:
        adapters.printer.add_object.assert_not_called()
    assert adapters.printer.lookup_object("probe_list", None) is None


@pytest.mark.parametrize(
    "error", [ImportError("broken probe import"), ModuleNotFoundError("missing dependency", name="dependency")]
)
def test_import_failures_propagate(mocker: MockerFixture, adapters: Mock, error: ImportError) -> None:
    _ = mocker.patch("cartographer.adapters.kalico.integrator.import_module", side_effect=error)

    with pytest.raises(ImportError) as caught:
        _ = KalicoIntegrator(adapters)

    assert caught.value is error
    adapters.printer.add_object.assert_not_called()


def test_duplicate_name_error_propagates(adapters: Mock, cartographer: Mock, registry_module: ModuleType) -> None:
    del registry_module
    cartographer.config.general.register_as_probe = False
    integrator = KalicoIntegrator(adapters)
    integrator.register_probe(cartographer)
    error = ValueError("duplicate cartographer")
    adapters.config.wrapper.error.side_effect = None
    adapters.config.wrapper.error.return_value = error

    with pytest.raises(ValueError) as caught:
        integrator.register_probe(cartographer)

    assert caught.value is error
    adapters.config.wrapper.error.assert_called_once_with("Duplicate probe name cartographer")
    adapters.printer.add_object.assert_called_once()


def test_conflicting_default_config_error_propagates(
    adapters: Mock, cartographer: Mock, registry_module: ModuleType
) -> None:
    del registry_module
    error = ValueError("conflicting default")

    def has_section(name: str) -> bool:
        return name == "probe"

    adapters.config.wrapper.has_section.side_effect = has_section
    adapters.config.wrapper.error.side_effect = None
    adapters.config.wrapper.error.return_value = error

    with pytest.raises(ValueError) as caught:
        KalicoIntegrator(adapters).register_probe(cartographer)

    assert caught.value is error
    adapters.printer.add_object.assert_called_once()


PROBE_COMMANDS = ("PROBE", "PROBE_ACCURACY", "QUERY_PROBE", "Z_OFFSET_APPLY_PROBE")


@pytest.fixture
def runnable_cartographer(adapters: Mock, config: Configuration, is_default: bool) -> PrinterCartographer:
    config.general = replace(config.general, register_as_probe=is_default)
    config.scan = replace(config.scan, models={})
    core_adapters = Mock(config=config, mcu=adapters.mcu, toolhead=adapters.toolhead, axis_twist_compensation=None)
    adapters.toolhead.get_position.return_value = Position(10, 20, 5)
    return PrinterCartographer(core_adapters)


@pytest.mark.parametrize("is_default", [True, False])
@pytest.mark.parametrize("selector", ["cartographer", None])
@pytest.mark.parametrize("name", PROBE_COMMANDS)
def test_mux_executes_existing_macros(
    mocker: MockerFixture,
    adapters: Mock,
    runnable_cartographer: PrinterCartographer,
    registry_module: ModuleType,
    is_default: bool,
    selector: str | None,
    name: str,
) -> None:
    del registry_module
    device = runnable_cartographer
    scan = mocker.patch("cartographer.probe.probe.Probe.perform_scan", return_value=2.5)
    query = mocker.patch("cartographer.probe.probe.Probe.query_is_triggered", return_value=True)
    model_config = ScanModelConfiguration("default", [1.0], (1.0, 2.0), 1.0, 25.0)
    model = Mock(z_offset=1.0, config=model_config)
    _ = mocker.patch.object(device.scan_mode, "get_model", return_value=model)
    adapters.toolhead.get_gcode_z_offset.return_value = 0.2
    integrator = KalicoIntegrator(adapters)
    integrator.register_probe(device)
    if is_default:
        for registration in device.macros:
            integrator.register_macro(registration)
    gcode = adapters.printer.lookup_object("gcode")
    assert set(gcode.mux_commands) == set(PROBE_COMMANDS)
    assert set(gcode.mux_commands[name][1]) == ({"cartographer", None} if is_default else {"cartographer"})
    assert gcode.descriptions[name] == next(reg.macro.description for reg in device.probe_macros if reg.name == name)
    params = {"PROBE": selector} if selector is not None else {}
    if name == "PROBE_ACCURACY":
        params.update(SAMPLES="3", LIFT_SPEED="7", SAMPLE_RETRACT_DIST="2")
    original = params.copy()
    for unknown in ("unknown", "Cartographer", "CARTOGRAPHER"):
        with pytest.raises(ValueError, match="Invalid PROBE"):
            gcode.dispatch(name, {"PROBE": unknown})
    scan.assert_not_called()
    query.assert_not_called()
    adapters.toolhead.move.assert_not_called()
    assert not gcode.clones
    assert not device.config.scan.models
    if selector is None and not is_default:
        with pytest.raises(ValueError, match="Missing PROBE"):
            gcode.dispatch(name, params)
        scan.assert_not_called()
        query.assert_not_called()
        adapters.toolhead.move.assert_not_called()
        assert not gcode.clones
        return
    command = gcode.dispatch(name, params)
    assert params == original
    clone = gcode.clones[-1]
    assert clone is not command
    assert clone.get_command() == command.get_command()
    assert clone.get_commandline() == command.get_commandline()
    assert clone.params == {key: value for key, value in original.items() if key != "PROBE"}
    assert clone.params is not params
    if name == "PROBE":
        scan.assert_called_once_with()
        assert device.probe_macro.last_trigger_position == 2.5
    elif name == "QUERY_PROBE":
        query.assert_called_once_with()
        assert device.query_probe_macro.last_triggered is True
    elif name == "PROBE_ACCURACY":
        assert scan.call_count == 3
        assert adapters.toolhead.move.call_count == 4
        adapters.toolhead.move.assert_called_with(z=7, speed=7)
    else:
        assert device.config.scan.models["default"].z_offset == 0.8


@pytest.mark.parametrize("other_first", [True, False])
def test_nondefault_preserves_other_probe(
    adapters: Mock,
    cartographer: Mock,
    registry_module: ModuleType,
    other_first: bool,
) -> None:
    del registry_module
    cartographer.config.general.register_as_probe = False
    handlers = {name: Mock() for name in PROBE_COMMANDS}
    cartographer.probe_macros = [MacroRegistration(name, handler) for name, handler in handlers.items()]
    gcode = adapters.printer.lookup_object("gcode")
    other = Mock()

    def register_other() -> None:
        for name in PROBE_COMMANDS:
            gcode.register_mux_command(name, "PROBE", "other", other)
            gcode.register_mux_command(name, "PROBE", None, other)

    if other_first:
        register_other()
    KalicoIntegrator(adapters).register_probe(cartographer)
    if not other_first:
        register_other()
    for name, macro in handlers.items():
        gcode.dispatch(name, {})
        gcode.dispatch(name, {"PROBE": "other"})
        assert other.call_count == 2
        other.reset_mock()
        gcode.dispatch(name, {"PROBE": "cartographer", "EXTRA": "untouched"})
        macro.run.assert_called_once()
        assert macro.run.call_args.args[0].params == {"EXTRA": "untouched"}
        macro.run.reset_mock()
        for selector in ("unknown", "Cartographer", "CARTOGRAPHER"):
            with pytest.raises(ValueError, match="Invalid PROBE"):
                gcode.dispatch(name, {"PROBE": selector})
        macro.run.assert_not_called()
        other.assert_not_called()


@pytest.mark.parametrize(
    ("registry_active", "name"),
    [(True, "CARTOGRAPHER_SCAN_PROBE"), (True, "PROBE_CALIBRATE")]
    + [(False, name) for name in (*PROBE_COMMANDS, "CARTOGRAPHER_SCAN_PROBE", "PROBE_CALIBRATE")],
)
def test_direct_registration_unchanged(
    mocker: MockerFixture,
    adapters: Mock,
    cartographer: Mock,
    registry_module: ModuleType,
    registry_active: bool,
    name: str,
) -> None:
    del registry_module
    if not registry_active:
        _ = mocker.patch(
            "cartographer.adapters.kalico.integrator.import_module",
            side_effect=ModuleNotFoundError(name="extras.probe"),
        )
    integrator = KalicoIntegrator(adapters)
    integrator.register_probe(cartographer)
    macro = Mock()
    integrator.register_macro(MacroRegistration(name, macro))
    gcode = adapters.printer.lookup_object("gcode")
    assert not gcode.mux_commands
    gcmd = gcode.dispatch(name, {"PROBE": "retained"})
    macro.run.assert_called_once_with(gcmd)
    assert not gcode.clones


def test_registry_detection_alone_does_not_enable_mux(
    adapters: Mock,
    registry_module: ModuleType,
) -> None:
    del registry_module
    macro = Mock()
    integrator = KalicoIntegrator(adapters)
    integrator.register_macro(MacroRegistration("PROBE", macro))
    gcode = adapters.printer.lookup_object("gcode")
    command = gcode.dispatch("PROBE", {"PROBE": "retained"})
    macro.run.assert_called_once_with(command)
    assert not gcode.mux_commands
    assert not gcode.clones


def test_mux_wraps_macro_errors(
    adapters: Mock,
    cartographer: Mock,
    registry_module: ModuleType,
) -> None:
    del registry_module
    integrator = KalicoIntegrator(adapters)
    integrator.register_probe(cartographer)
    macro = Mock()
    macro.run.side_effect = RuntimeError("macro failure")
    integrator.register_macro(MacroRegistration("PROBE", macro))
    with pytest.raises(ValueError, match="macro failure"):
        adapters.printer.lookup_object("gcode").dispatch("PROBE", {"PROBE": "cartographer"})


def test_existing_default_alias_error_propagates(
    adapters: Mock, cartographer: Mock, registry_module: ModuleType
) -> None:
    del registry_module
    existing_probe = object()
    adapters.printer.add_object("probe", existing_probe)

    with pytest.raises(ValueError, match="Duplicate printer object probe"):
        KalicoIntegrator(adapters).register_probe(cartographer)

    assert adapters.printer.lookup_object("probe") is existing_probe
    assert adapters.printer.lookup_object("probe_list").get_all() == {}


@pytest.fixture
def mesh_routing(
    mocker: MockerFixture,
    adapters: Mock,
    cartographer: Mock,
    registry_module: ModuleType,
    is_default: bool,
) -> tuple[BedMeshCalibrateMacro, Mock, Mock]:
    del registry_module
    cartographer.config.general.register_as_probe = is_default
    integrator = KalicoIntegrator(adapters)
    integrator.register_probe(cartographer)
    macro = BedMeshCalibrateMacro(Mock(), adapters.toolhead, Mock(), None, Mock(), Mock())
    scan = Mock()
    original_run = macro.run

    def run(command: GCodeCommand) -> None:
        if command.get("METHOD", "scan").lower() == "scan":
            scan(command)
        else:
            original_run(command)

    _ = mocker.patch.object(macro, "run", side_effect=run)
    _ = mocker.patch("cartographer.adapters.klipper_like.integrator.GCodeCommand", GCodeCommand)
    native = Mock()
    gcode = adapters.printer.lookup_object("gcode")
    _ = gcode.register_command("BED_MESH_CALIBRATE", native)
    integrator.register_macro(MacroRegistration("BED_MESH_CALIBRATE", macro))
    assert gcode.removals == ["BED_MESH_CALIBRATE"]
    return macro, scan, native


@pytest.mark.parametrize(("is_default", "selector"), [(True, None), (True, "cartographer"), (False, "cartographer")])
@pytest.mark.parametrize("method", [None, "scan", "SCAN"])
def test_mesh_cartographer_scan_consumes_only_selector(
    adapters: Mock,
    mesh_routing: tuple[BedMeshCalibrateMacro, Mock, Mock],
    is_default: bool,
    selector: str | None,
    method: str | None,
) -> None:
    del is_default
    macro, scan, native = mesh_routing
    gcode = adapters.printer.lookup_object("gcode")
    registry = adapters.printer.lookup_object("probe_list")
    params = {"PROFILE": "selected"}
    if selector is not None:
        params["PROBE"] = selector
    if method is not None:
        params["METHOD"] = method
    original = params.copy()

    command = gcode.dispatch("BED_MESH_CALIBRATE", params)

    clone = gcode.clones[-1]
    assert clone is not command
    assert clone.params is not params
    assert clone.params == {key: value for key, value in original.items() if key != "PROBE"}
    assert clone.command == command.command
    assert clone.commandline == command.commandline
    assert command.params is params
    assert params == original
    assert registry.selections == [(command, None)]
    cast("Mock", macro.run).assert_called_once_with(clone)
    assert scan.call_args.args[0] is clone
    native.assert_not_called()
    adapters.toolhead.move.assert_not_called()


@pytest.mark.parametrize("is_default", [True, False])
def test_mesh_cartographer_automatic_fallback_retains_original(
    adapters: Mock, mesh_routing: tuple[BedMeshCalibrateMacro, Mock, Mock], is_default: bool
) -> None:
    del is_default
    macro, scan, native = mesh_routing
    gcode = adapters.printer.lookup_object("gcode")
    registry = adapters.printer.lookup_object("probe_list")
    selected: list[object | None] = []

    def native_run(command: GCodeCommand) -> None:
        selected.append(registry.get_command_probe(command))

    native.side_effect = native_run
    params = {"PROBE": "cartographer", "METHOD": "automatic", "PROFILE": "retained"}
    original = params.copy()

    command = gcode.dispatch("BED_MESH_CALIBRATE", params)

    assert cast("Mock", macro.run).call_args.args[0] is command
    assert native.call_args.args[0] is command
    assert selected == [registry.probes["cartographer"]]
    assert selected[0] is registry.probes["cartographer"]
    assert command.params is params
    assert params == original
    assert not gcode.clones
    scan.assert_not_called()


@pytest.mark.parametrize("is_default", [False])
@pytest.mark.parametrize("selector", [None, "other"])
@pytest.mark.parametrize("method", [None, "automatic", "manual"])
def test_mesh_other_probe_delegates_untouched(
    adapters: Mock,
    mesh_routing: tuple[BedMeshCalibrateMacro, Mock, Mock],
    is_default: bool,
    selector: str | None,
    method: str | None,
) -> None:
    del is_default
    macro, scan, native = mesh_routing
    gcode = adapters.printer.lookup_object("gcode")
    registry = adapters.printer.lookup_object("probe_list")
    other = object()
    registry.probes["other"] = other
    registry.default_probe = other
    params = {"PROFILE": "native"}
    if selector is not None:
        params["PROBE"] = selector
    if method is not None:
        params["METHOD"] = method
    original = params.copy()
    selected: list[object | None] = []

    def native_run(command: GCodeCommand) -> None:
        selected.append(registry.get_command_probe(command))

    native.side_effect = native_run

    command = gcode.dispatch("BED_MESH_CALIBRATE", params)

    assert native.call_args.args[0] is command
    assert selected[0] is other
    assert command.params is params
    assert params == original
    assert not gcode.clones
    cast("Mock", macro.run).assert_not_called()
    scan.assert_not_called()
    adapters.toolhead.move.assert_not_called()


@pytest.mark.parametrize("is_default", [False])
@pytest.mark.parametrize(
    ("params", "error"),
    [
        ({"PROBE": "other", "METHOD": "scan"}, "scan.*Cartographer"),
        ({"PROBE": "unknown"}, "Unknown requested probe unknown"),
        ({"PROBE": "unknown", "METHOD": "manual"}, "Unknown requested probe unknown"),
        ({}, "No default probe"),
        ({"METHOD": "scan"}, "No default probe"),
        ({"METHOD": "automatic"}, "No default probe"),
    ],
)
def test_mesh_selection_errors_precede_work(
    adapters: Mock,
    mesh_routing: tuple[BedMeshCalibrateMacro, Mock, Mock],
    is_default: bool,
    params: dict[str, str],
    error: str,
) -> None:
    del is_default
    macro, scan, native = mesh_routing
    registry = adapters.printer.lookup_object("probe_list")
    registry.probes["other"] = object()
    gcode = adapters.printer.lookup_object("gcode")
    original = params.copy()

    with pytest.raises(ValueError, match=error):
        gcode.dispatch("BED_MESH_CALIBRATE", params)

    assert params == original
    if "PROBE" not in params:
        assert [default for _, default in registry.selections] == [None, _MISSING_DEFAULT]
        assert registry.selections[0][0] is registry.selections[1][0]
    cast("Mock", macro.run).assert_not_called()
    native.assert_not_called()
    scan.assert_not_called()
    assert not gcode.clones
    adapters.toolhead.move.assert_not_called()
    cast("Mock", macro.adapter).clear_mesh.assert_not_called()
    cast("Mock", macro.adapter).apply_mesh.assert_not_called()


@pytest.mark.parametrize("is_default", [False])
@pytest.mark.parametrize("selector", [None, "cartographer", "other"])
def test_mesh_manual_without_default_validates_then_delegates(
    adapters: Mock,
    mesh_routing: tuple[BedMeshCalibrateMacro, Mock, Mock],
    is_default: bool,
    selector: str | None,
) -> None:
    del is_default
    macro, scan, native = mesh_routing
    registry = adapters.printer.lookup_object("probe_list")
    registry.probes["other"] = object()
    gcode = adapters.printer.lookup_object("gcode")
    params = {"METHOD": "manual"}
    if selector is not None:
        params["PROBE"] = selector
    original = params.copy()

    command = gcode.dispatch("BED_MESH_CALIBRATE", params)

    assert registry.selections == [(command, None)]
    assert native.call_args.args[0] is command
    assert command.params is params
    assert params == original
    cast("Mock", macro.run).assert_not_called()
    scan.assert_not_called()
    assert not gcode.clones


@pytest.mark.parametrize(
    ("is_default", "params", "error"),
    [
        (True, {"METHOD": "manual"}, "native BED_MESH_CALIBRATE.*unavailable"),
        (True, {"PROBE": "other"}, "native BED_MESH_CALIBRATE.*unavailable"),
        (True, {"PROBE": "other", "METHOD": "automatic"}, "native BED_MESH_CALIBRATE.*unavailable"),
        (True, {"PROBE": "cartographer", "METHOD": "automatic"}, "native BED_MESH_CALIBRATE.*unavailable"),
        (False, {}, "No default probe"),
        (False, {"METHOD": "scan"}, "No default probe"),
        (False, {"METHOD": "automatic"}, "No default probe"),
    ],
)
def test_mesh_missing_native_handler_errors_before_work(
    adapters: Mock,
    cartographer: Mock,
    registry_module: ModuleType,
    is_default: bool,
    params: dict[str, str],
    error: str,
) -> None:
    del registry_module
    cartographer.config.general.register_as_probe = is_default
    integrator = KalicoIntegrator(adapters)
    integrator.register_probe(cartographer)
    registry = adapters.printer.lookup_object("probe_list")
    registry.probes["other"] = object()
    macro = Mock()
    integrator.register_macro(MacroRegistration("BED_MESH_CALIBRATE", macro))
    gcode = adapters.printer.lookup_object("gcode")
    original = params.copy()

    with pytest.raises(ValueError, match=error):
        gcode.dispatch("BED_MESH_CALIBRATE", params)

    assert params == original
    assert gcode.removals == ["BED_MESH_CALIBRATE"]
    assert not gcode.clones
    cast("Mock", macro.run).assert_not_called()
    adapters.toolhead.move.assert_not_called()


@pytest.mark.parametrize("method", [None, "scan", "automatic", "manual"])
def test_legacy_mesh_routing_unchanged(
    mocker: MockerFixture, adapters: Mock, cartographer: Mock, method: str | None
) -> None:
    _ = mocker.patch(
        "cartographer.adapters.kalico.integrator.import_module",
        side_effect=ModuleNotFoundError(name="extras.probe"),
    )
    _ = mocker.patch("cartographer.adapters.klipper_like.integrator.GCodeCommand", GCodeCommand)
    integrator = KalicoIntegrator(adapters)
    integrator.register_probe(cartographer)
    macro = BedMeshCalibrateMacro(Mock(), adapters.toolhead, Mock(), None, Mock(), Mock())
    scan = mocker.patch(
        "cartographer.macros.bed_mesh.scan_mesh.MeshScanParams.from_macro_params",
        side_effect=RuntimeError("legacy scan reached"),
    )
    native = Mock()
    gcode = adapters.printer.lookup_object("gcode")
    _ = gcode.register_command("BED_MESH_CALIBRATE", native)
    integrator.register_macro(MacroRegistration("BED_MESH_CALIBRATE", macro))
    params = {"PROBE": "untouched"}
    if method is not None:
        params["METHOD"] = method
    original = params.copy()
    if method in (None, "scan"):
        with pytest.raises(ValueError, match="legacy scan reached"):
            gcode.dispatch("BED_MESH_CALIBRATE", params)
        scan.assert_called_once()
        native.assert_not_called()
    else:
        command = gcode.dispatch("BED_MESH_CALIBRATE", params)
        assert native.call_args.args[0] is command
        assert command.params is params
        scan.assert_not_called()
    assert params == original
    assert not gcode.clones


@pytest.mark.parametrize("fails", [False, True])
def test_mesh_named_scan_without_native_handler(
    adapters: Mock, cartographer: Mock, registry_module: ModuleType, fails: bool
) -> None:
    del registry_module
    cartographer.config.general.register_as_probe = False
    integrator = KalicoIntegrator(adapters)
    integrator.register_probe(cartographer)
    macro = Mock()
    integrator.register_macro(MacroRegistration("BED_MESH_CALIBRATE", macro))
    gcode = adapters.printer.lookup_object("gcode")
    params = {"PROBE": "cartographer", "METHOD": "scan"}
    original = params.copy()
    if fails:
        macro.run.side_effect = RuntimeError("scan failed")
        with pytest.raises(ValueError, match="scan failed"):
            gcode.dispatch("BED_MESH_CALIBRATE", params)
    else:
        command = gcode.dispatch("BED_MESH_CALIBRATE", params)
        assert macro.run.call_args.args[0] is not command
    assert macro.run.call_args.args[0] is gcode.clones[-1]
    assert gcode.clones[-1].params == {"METHOD": "scan"}
    assert params == original
    assert gcode.removals == ["BED_MESH_CALIBRATE"]
    macro.set_fallback_macro.assert_not_called()


@pytest.mark.parametrize(("is_default", "selector"), [(True, None), (True, "cartographer"), (False, "cartographer")])
def test_native_point_consumers_and_rapid_scan_fallback(
    mocker: MockerFixture,
    adapters: Mock,
    runnable_cartographer: PrinterCartographer,
    registry_module: ModuleType,
    is_default: bool,
    selector: str | None,
) -> None:
    del registry_module
    device = runnable_cartographer
    integrator = KalicoIntegrator(adapters)
    before = adapters.mcu.mock_calls.copy()
    integrator.register_probe(device)
    assert adapters.mcu.mock_calls == before
    registry = adapters.printer.lookup_object("probe_list")
    probe = registry.probes["cartographer"]
    assert registry.get_default_probe() is (probe if is_default else None)
    assert type(probe.mcu_probe) is KlipperEndstop
    assert probe.mcu_probe.mcu is adapters.mcu
    assert probe.mcu_probe.endstop is device.scan_mode
    assert not hasattr(probe.mcu_probe, "begin_collect_distance")
    params = {"METHOD": "rapid_scan", "LIFT_SPEED": "8"}
    if selector is not None:
        params["PROBE"] = selector
    command = GCodeCommand("BED_MESH_CALIBRATE", "rapid_scan", params)
    scan = mocker.patch.object(device.scan_mode, "perform_probe", side_effect=[1.5, 2.5, 3.5])
    lift = mocker.spy(probe, "get_lift_speed")
    offsets = mocker.spy(probe, "get_offsets")
    begin = mocker.spy(probe, "multi_probe_begin")
    run = mocker.spy(probe, "run_probe")
    end = mocker.spy(probe, "multi_probe_end")
    retry_session = object()

    def before_automatic(message: str) -> None:
        assert message == "METHOD=rapid_scan not supported for probe cartographer, using automatic"
        for call in (lift, offsets, begin, run, end, scan):
            call.assert_not_called()
        adapters.toolhead.move.assert_not_called()

    command.respond_info.side_effect = before_automatic

    def native_start_probe(gcmd: GCodeCommand) -> list[list[float]]:
        # Minimal start_probe consumer contract from Kalico extras/probe.py:1146.
        selected = registry.get_command_probe(gcmd, None)
        method = gcmd.get("METHOD", "automatic").lower()
        if selected is not None and method == "rapid_scan":
            mcu_probe = selected.mcu_probe
            can_scan = hasattr(mcu_probe, "begin_collect_distance")
            if can_scan:
                message = "Unexpected native rapid-scan capability"
                raise AssertionError(message)
            else:
                gcmd.respond_info(f"METHOD=rapid_scan not supported for probe {selected.name}, using automatic")
                method = "automatic"
        assert selected is probe
        assert method == "automatic"
        assert selected.get_lift_speed(gcmd) == 8
        assert selected.get_offsets() == device.scan_mode.offset.as_tuple()
        selected.multi_probe_begin()
        positions: list[list[float]] = []
        for x, y in [(10, 20), (30, 40)]:
            adapters.toolhead.get_position.return_value = Position(x, y, 5)
            positions.append(selected.run_probe(gcmd, retry_session))
        selected.multi_probe_end()
        return positions

    assert native_start_probe(command) == [[10, 20, 1.5], [30, 40, 2.5]]
    assert registry.selections == [(command, None)]
    command.respond_info.assert_called_once_with(
        "METHOD=rapid_scan not supported for probe cartographer, using automatic"
    )
    lift.assert_called_once_with(command)
    offsets.assert_called_once_with()
    begin.assert_called_once_with()
    assert run.call_args_list == [((command, retry_session),), ((command, retry_session),)]
    end.assert_called_once_with()
    assert probe.probe_name == "cartographer"
    assert probe.get_lift_speed() == device.config.general.lift_speed
    # Axis twist consumes the same single-probe position semantics.
    assert probe.run_probe(command) == [30, 40, 3.5]
    assert scan.call_count == 3
    measure = mocker.patch.object(device.scan_mode, "measure_distance", return_value=0.5)
    has_model = mocker.patch.object(device.scan_mode, "has_model", return_value=False)
    assert probe.mcu_probe.query_endstop(12.5) == 1
    measure.assert_not_called()
    has_model.return_value = True
    assert probe.mcu_probe.query_endstop(13.5) == 1
    measure.assert_called_once_with(time=13.5)
    measure.return_value = device.scan_mode.get_endstop_position() + 1
    assert probe.mcu_probe.query_endstop(14.5) == 0
    measure.assert_called_with(time=14.5)


@final
class NativeZCalibration:
    """Native commands (including aliases) load the selected probe before motion."""

    def __init__(self, registry: ProbeList) -> None:
        self.registry = registry
        self.loaded: list[object] = []
        self.motion = Mock()
        self.result = object()

    def load_probe(self, selected: object) -> object:
        self.loaded.append(selected)
        return self.result

    def command(self, gcmd: GCodeCommand) -> None:
        _ = self.load_probe(self.registry.get_command_probe(gcmd))
        self.motion()


@pytest.mark.parametrize("is_default", [True, False])
@pytest.mark.parametrize(
    "entry", ["CALIBRATE_Z", "PROBE_Z_ACCURACY", "CALIBRATE_Z_BASE", "PROBE_Z_ACCURACY_BASE", "helper"]
)
@pytest.mark.parametrize("selector", [None, "cartographer", "other", "other_default", "unknown"])
def test_native_z_calibration_boundary(
    mocker: MockerFixture,
    adapters: Mock,
    cartographer: Mock,
    registry_module: ModuleType,
    is_default: bool,
    entry: str,
    selector: str | None,
) -> None:
    del registry_module

    def has_section(name: str) -> bool:
        return name == "z_calibration"

    adapters.config.wrapper.has_section.side_effect = has_section
    adapters.printer.command_error = ValueError
    cartographer.config.general.register_as_probe = is_default
    integrator = KalicoIntegrator(adapters)
    _ = mocker.patch.object(integrator, "_configure_macro_logger")
    integrator.setup()
    integrator.register_probe(cartographer)
    registry = adapters.printer.lookup_object("probe_list")
    other = object()
    registry.probes["other"] = other
    if selector == "other_default":
        registry.default_probe = other
    # The native helper can be loaded after Cartographer setup/registration.
    helper = NativeZCalibration(registry)
    original = mocker.spy(helper, "load_probe")
    gcode = adapters.printer.lookup_object("gcode")
    if entry != "helper":
        _ = gcode.register_command(entry, helper.command)
    adapters.printer.add_object("z_calibration", helper)
    callbacks = [
        call.args[1]
        for call in adapters.printer.register_event_handler.call_args_list
        if call.args[0] == "klippy:connect"
    ]
    assert len(callbacks) == 1
    callbacks[0]()
    guarded = helper.load_probe
    params = {} if selector in (None, "other_default") else {"PROBE": selector}
    command = GCodeCommand(entry, entry, params)

    def invoke() -> object:
        if entry == "helper":
            return guarded(registry.get_command_probe(command))
        return gcode.dispatch(entry, params)

    if selector == "unknown" or (selector is None and not is_default):
        with pytest.raises(ValueError, match="Unknown requested probe|No default probe"):
            _ = invoke()
        original.assert_not_called()
        helper.motion.assert_not_called()
    elif selector not in ("other", "other_default"):
        with pytest.raises(ValueError, match="native Z calibration.*unsupported.*Cartographer calibration"):
            _ = invoke()
        original.assert_not_called()
        helper.motion.assert_not_called()
    else:
        result = invoke()
        original.assert_called_once_with(other)
        assert helper.loaded == [other]
        if entry == "helper":
            assert result is helper.result
            helper.motion.assert_not_called()
        else:
            helper.motion.assert_called_once_with()


@pytest.mark.parametrize("bad_helper", [None, object(), Mock(load_probe=None)])
def test_missing_native_z_calibration_helper_is_config_error(
    mocker: MockerFixture, adapters: Mock, cartographer: Mock, registry_module: ModuleType, bad_helper: object
) -> None:
    del registry_module

    def has_section(name: str) -> bool:
        return name == "z_calibration"

    adapters.config.wrapper.has_section.side_effect = has_section
    integrator = KalicoIntegrator(adapters)
    _ = mocker.patch.object(integrator, "_configure_macro_logger")
    integrator.setup()
    integrator.register_probe(cartographer)
    if bad_helper is not None:
        adapters.printer.add_object("z_calibration", bad_helper)
    callback = next(
        call.args[1]
        for call in adapters.printer.register_event_handler.call_args_list
        if call.args[0] == "klippy:connect"
    )
    with pytest.raises(ValueError, match="z_calibration.*load_probe"):
        callback()
    adapters.toolhead.move.assert_not_called()


@pytest.mark.parametrize("registry_active", [True, False])
def test_native_safety_inactive_without_section_or_registry(
    mocker: MockerFixture, adapters: Mock, cartographer: Mock, registry_module: ModuleType, registry_active: bool
) -> None:
    del registry_module
    if not registry_active:
        _ = mocker.patch(
            "cartographer.adapters.kalico.integrator.import_module",
            side_effect=ModuleNotFoundError(name="extras.probe"),
        )
        adapters.config.wrapper.has_section.return_value = True
    integrator = KalicoIntegrator(adapters)
    _ = mocker.patch.object(integrator, "_configure_macro_logger")
    integrator.setup()
    integrator.register_probe(cartographer)
    helper = NativeZCalibration(ProbeList())
    original = helper.load_probe
    adapters.printer.add_object("z_calibration", helper)
    callbacks = [
        call.args[1]
        for call in adapters.printer.register_event_handler.call_args_list
        if call.args[0] == "klippy:connect"
    ]
    if registry_active:
        assert len(callbacks) == 1
        callbacks[0]()
    else:
        assert not callbacks
        adapters.config.wrapper.has_section.assert_not_called()
    assert helper.load_probe == original


@pytest.mark.parametrize("is_default", [True, False])
@pytest.mark.parametrize("selector", [None, "cartographer", "other", "Cartographer"])
@pytest.mark.parametrize("overrides", [False, True])
def test_nozzle_cleanup_selector_before_registration(
    adapters: Mock,
    cartographer: Mock,
    registry_module: ModuleType,
    is_default: bool,
    selector: str | None,
    overrides: bool,
) -> None:
    del registry_module
    cartographer.config.general.register_as_probe = is_default

    def has_section(name: str) -> bool:
        return name == "nozzle_cleanup"

    adapters.config.wrapper.has_section.side_effect = has_section
    section = adapters.config.wrapper.getsection.return_value
    section.get.return_value = selector
    section.error.side_effect = adapters.config.wrapper.error.side_effect
    if overrides:
        section.getfloat.return_value = 7.0
    integrator = KalicoIntegrator(adapters)
    if selector == "cartographer" or (selector is None and is_default):
        with pytest.raises(ValueError, match="nozzle_cleanup.*unsupported.*Cartographer"):
            integrator.register_probe(cartographer)
        assert adapters.printer.lookup_object("probe_list") is None
        adapters.printer.add_object.assert_not_called()
    else:
        integrator.register_probe(cartographer)
        assert "cartographer" in adapters.printer.lookup_object("probe_list").probes
    section.get.assert_called_once_with("probe", None)
    section.getfloat.assert_not_called()
