from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING
from unittest.mock import Mock, call

import pytest

from cartographer.adapters.kalico.axis_twist_compensation import (
    KalicoAxisTwistCompensationAdapter,
    KalicoMeshAxisTwistCompensationAdapter,
    create_axis_twist_compensation,
)
from cartographer.core import PrinterCartographer
from cartographer.interfaces.printer import MacroParams
from cartographer.macros.axis_twist_compensation import AxisTwistCompensationAdapter
from cartographer.macros.message import MessageMacro

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

    from cartographer.interfaces.configuration import Configuration


@pytest.fixture
def host(mocker: MockerFixture) -> Mock:
    host = Mock()
    host.config.get_printer.return_value = host.printer
    host.config.has_section.return_value = True
    host.config.get_prefix_sections.return_value = []
    host.registry.get_probe.return_value = None
    # Reject every legacy movement/calibration attribute, including writes.
    host.manager = Mock(spec_set=["get_z_compensation_value"])
    host.manager.get_z_compensation_value.return_value = 0.125
    host.api.get_list.return_value = host.registry
    host.native = ModuleType("extras.axis_twist_compensation")
    host.native.AxisTwistCompensation = type("NativeManager", (), {"get_compensation": Mock()})
    probe_module = ModuleType("extras.probe")
    probe_module.__dict__["ProbeList"] = host.api
    modules = {"extras.axis_twist_compensation": host.native, "extras.probe": probe_module}
    host.importer = mocker.patch(
        "cartographer.adapters.kalico.axis_twist_compensation.import_module", side_effect=modules.__getitem__
    )

    def load(config: object, name: str) -> Mock:
        assert config is host.config
        assert name == "axis_twist_compensation"
        host.api.get_list.assert_called_once_with(host.printer)
        return host.manager

    host.printer.load_object.side_effect = load
    return host


@pytest.mark.parametrize("capability", ["missing", "noncallable", "callable"])
def test_selects_class_capability_before_legacy_initialization(
    mocker: MockerFixture, host: Mock, capability: str
) -> None:
    if capability == "missing":
        del host.native.AxisTwistCompensation.get_compensation
    elif capability == "noncallable":
        host.native.AxisTwistCompensation.get_compensation = None
    legacy_init = mocker.patch.object(KalicoAxisTwistCompensationAdapter, "__init__", return_value=None)
    adapter = create_axis_twist_compensation(host.config)
    if capability == "callable":
        assert type(adapter) is KalicoMeshAxisTwistCompensationAdapter
        legacy_init.assert_not_called()
        host.printer.lookup_object.assert_not_called()
    else:
        assert type(adapter) is KalicoAxisTwistCompensationAdapter
        legacy_init.assert_called_once_with(host.config)
        host.api.get_list.assert_not_called()
        host.printer.load_object.assert_not_called()


def test_legacy_adapter_retains_calibration_and_lookup(host: Mock) -> None:
    manager = Mock(spec=["horizontal_move_z", "speed", "get_z_compensation_value"])
    manager.horizontal_move_z = 5.0
    manager.speed = 40.0
    manager.get_z_compensation_value.return_value = 0.25
    host.printer.load_object.side_effect = None
    host.printer.load_object.return_value = manager
    adapter = KalicoAxisTwistCompensationAdapter(host.config)
    assert adapter.move_height == 5.0
    assert adapter.speed == 40.0
    assert adapter.available_axes == ("x", "y")
    assert isinstance(adapter, AxisTwistCompensationAdapter)
    assert adapter.get_z_compensation_value(x=12.0, y=34.0) == 0.25
    manager.get_z_compensation_value.assert_called_once_with([12.0, 34.0, 0])


@pytest.mark.parametrize("default", ["other", "cartographer", None])
def test_native_lookup_resolves_late_and_passes_exact_registered_probe(host: Mock, default: str | None) -> None:
    # No probe exists during core construction, regardless of the eventual default.
    adapter = create_axis_twist_compensation(host.config)
    host.registry.get_probe.assert_not_called()
    with pytest.raises(RuntimeError, match="Probe 'cartographer' is not registered"):
        _ = adapter.get_z_compensation_value(x=12.0, y=34.0)
    host.manager.get_z_compensation_value.assert_not_called()

    probe = Mock(spec=["probe_name", "get_offsets"])
    probe.probe_name = "cartographer"
    probe.get_offsets.return_value = (7.0, -9.0, 2.0)
    other = Mock()
    host.registry.get_default_probe.return_value = probe if default == "cartographer" else other if default else None
    host.registry.get_probe.side_effect = {"cartographer": probe, "other": other}.get
    assert adapter.get_z_compensation_value(x=12.0, y=34.0) == 0.125
    host.registry.get_probe.assert_called_with("cartographer")
    host.registry.get_default_probe.assert_not_called()
    probe.get_offsets.assert_not_called()
    host.manager.get_z_compensation_value.assert_called_once_with([12.0, 34.0, 0], probe=probe)
    assert host.manager.get_z_compensation_value.call_args.kwargs["probe"] is probe
    assert host.manager.mock_calls == [call.get_z_compensation_value([12.0, 34.0, 0], probe=probe)]


@pytest.mark.parametrize("probe_name", ["cartographer_touch", "missing"])
def test_explicit_native_probe_lookup(host: Mock, probe_name: str) -> None:
    adapter = KalicoMeshAxisTwistCompensationAdapter(host.config)
    probe = object()
    host.registry.get_probe.side_effect = {"cartographer_touch": probe}.get
    if probe_name == "missing":
        with pytest.raises(RuntimeError, match="Probe 'missing' is not registered"):
            _ = adapter.get_z_compensation_value(x=12, y=34, probe_name=probe_name)
        host.manager.get_z_compensation_value.assert_not_called()
    else:
        assert adapter.get_z_compensation_value(x=12, y=34, probe_name=probe_name) == 0.125
        host.manager.get_z_compensation_value.assert_called_once_with([12, 34, 0], probe=probe)
    host.registry.get_probe.assert_called_once_with(probe_name)


@pytest.mark.parametrize("map_state", ["missing", "unknown", "loaded"])
def test_native_manager_owns_map_resolution_without_writes(host: Mock, map_state: str) -> None:
    probe = object()
    host.registry.get_probe.return_value = probe
    expected = 0.125 if map_state == "loaded" else 0.0
    host.manager.get_z_compensation_value.return_value = expected
    adapter = create_axis_twist_compensation(host.config)
    assert adapter.get_z_compensation_value(x=-13.0, y=47.0) == expected
    host.manager.get_z_compensation_value.assert_called_once_with([-13.0, 47.0, 0], probe=probe)
    # No configuration writes, calibration calls, or map mutations through the adapter.
    host.printer.lookup_object.assert_not_called()
    assert not isinstance(adapter, AxisTwistCompensationAdapter)
    assert not hasattr(adapter, "apply_compensation")
    assert not hasattr(adapter, "clear_compensations")
    assert not hasattr(adapter, "get_calibration_options")


def test_native_adapter_registers_guide_without_side_effects(host: Mock, config: Configuration) -> None:
    adapter = KalicoMeshAxisTwistCompensationAdapter(host.config)
    assert not isinstance(adapter, AxisTwistCompensationAdapter)
    adapters = Mock(config=config, axis_twist_compensation=adapter)
    cartographer = PrinterCartographer(adapters)
    macro = next(reg.macro for reg in cartographer.macros if reg.name == "CARTOGRAPHER_AXIS_TWIST_COMPENSATION")
    assert isinstance(macro, MessageMacro)
    calls = list(host.mock_calls)
    params = Mock(spec_set=MacroParams)
    with pytest.raises(RuntimeError, match="Use native AXIS_TWIST_COMPENSATION_CALIBRATE"):
        macro.run(params)
    assert host.mock_calls == calls
    assert params.mock_calls == []
    assert adapters.toolhead.mock_calls == []
    assert adapters.gcode.mock_calls == []


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("section", ["base", "named", "absent", "similar"])
def test_constructor_section_detection_and_fields(
    mocker: MockerFixture, host: Mock, native: bool, section: str
) -> None:
    mocker.patch.dict(
        "sys.modules",
        {"extras.homing": Mock(), "extras.manual_probe": Mock(), "extras.thermistor": Mock(), "greenlet": Mock()},
    )
    from cartographer.adapters.kalico.adapters import KalicoAdapters

    def has_section(name: str) -> bool:
        return section == "base" and name == "axis_twist_compensation"

    host.config.has_section.side_effect = has_section
    names = {
        "base": ["axis_twist_compensation"],
        "named": ["axis_twist_compensation cartographer"],
        "absent": [],
        "similar": ["axis_twist_compensation_other"],
    }[section]

    def get_prefix_sections(prefix: str) -> list[Mock]:
        return [Mock(get_name=Mock(return_value=name)) for name in names if name.startswith(prefix)]

    host.config.get_prefix_sections.side_effect = get_prefix_sections
    for name in (
        "parse",
        "KalicoMcuPlatform",
        "CartographerMcu",
        "KlipperConfiguration",
        "KalicoToolhead",
        "KlipperBedMesh",
        "KlipperGCodeDispatch",
        "KlipperScheduler",
    ):
        _ = mocker.patch(f"cartographer.adapters.kalico.adapters.{name}")
    legacy_init = mocker.patch.object(KalicoAxisTwistCompensationAdapter, "__init__", return_value=None)
    if not native:
        del host.native.AxisTwistCompensation.get_compensation
    adapters = KalicoAdapters(host.config)
    if section in ("absent", "similar"):
        assert adapters.axis_twist_compensation is None
        host.importer.assert_not_called()
        host.printer.load_object.assert_not_called()
        legacy_init.assert_not_called()
    elif native:
        assert type(adapters.axis_twist_compensation) is KalicoMeshAxisTwistCompensationAdapter
        assert not isinstance(adapters.axis_twist_compensation, AxisTwistCompensationAdapter)
        legacy_init.assert_not_called()
    else:
        assert type(adapters.axis_twist_compensation) is KalicoAxisTwistCompensationAdapter
        legacy_init.assert_called_once_with(host.config)
