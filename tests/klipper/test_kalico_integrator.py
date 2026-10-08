from __future__ import annotations

from dataclasses import replace
from types import ModuleType
from typing import TYPE_CHECKING, Callable, final
from unittest.mock import Mock

import pytest

from cartographer.adapters.kalico.integrator import KalicoIntegrator
from cartographer.adapters.kalico.probe import KalicoCartographerProbe
from cartographer.core import MacroRegistration, PrinterCartographer
from cartographer.extra import load_config
from cartographer.interfaces.configuration import Configuration, ScanModelConfiguration
from cartographer.interfaces.printer import Position
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

    def register_command(
        self, name: str, handler: Callable[[GCodeCommand], None] | None, desc: str | None = None
    ) -> Callable[[GCodeCommand], None] | None:
        if handler is None:
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


class ProbeList:
    """Registration-shaped registry; the registry alone owns the default alias."""

    def __init__(self) -> None:
        self.probes: dict[str, KalicoCartographerProbe] = {}
        self.default_probe: KalicoCartographerProbe | None = None
        self.configs: list[Mock] = []

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

    def get_all(self) -> dict[str, KalicoCartographerProbe]:
        return self.probes

    def get_default_probe(self) -> KalicoCartographerProbe | None:
        return self.default_probe


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
    host.config.wrapper.error.side_effect = ValueError
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
@pytest.mark.parametrize("missing", ["module", "class", "get_list", "add_probe_object"])
def test_missing_registry_uses_legacy_registration(
    mocker: MockerFixture, adapters: Mock, cartographer: Mock, is_default: bool, missing: str
) -> None:
    module = ModuleType("extras.probe")
    if missing in ("get_list", "add_probe_object"):
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
    adapters.config.wrapper.has_section.return_value = True
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
