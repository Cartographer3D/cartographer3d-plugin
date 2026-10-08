from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from cartographer.adapters.kalico.integrator import KalicoIntegrator
from cartographer.adapters.kalico.probe import KalicoCartographerProbe
from cartographer.extra import load_config

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


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
    objects: dict[str, object] = {"gcode": Mock()}

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
