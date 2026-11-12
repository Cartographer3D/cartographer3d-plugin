from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import Mock

import pytest

from cartographer.adapters.kalico.axis_twist_compensation import KalicoMeshAxisTwistCompensationAdapter
from cartographer.adapters.kalico.probe import KalicoTouchProbe
from cartographer.interfaces.printer import Position
from cartographer.probe.touch_mode import TouchBoundaries

if TYPE_CHECKING:
    from gcode import GCodeCommand
    from pytest_mock import MockerFixture


@pytest.fixture
def host(mocker: MockerFixture) -> Mock:
    host = Mock()
    _ = mocker.patch("cartographer.adapters.klipper_like.utils.CommandError", ValueError)
    host.config.get_printer.return_value = host.printer
    host.printer.load_object.return_value = host.manager
    host.api.ProbeList.get_list.return_value = host.registry
    _ = mocker.patch("cartographer.adapters.kalico.axis_twist_compensation.import_module", return_value=host.api)
    host.compensation = KalicoMeshAxisTwistCompensationAdapter(host.config)
    host.toolhead.get_position.return_value = Position(10, 20, 5)
    host.touch.offset = Position(0, 0, 0)
    host.touch.boundaries = TouchBoundaries(0, 100, 0, 100)
    host.touch.perform_probe.return_value = 1.5
    host.general.lift_speed = 6.0
    return host


def test_touch_uses_only_own_native_map_and_state(host: Mock) -> None:
    correction = 0.125
    probe = KalicoTouchProbe(host.toolhead, host.touch, host.general, host.compensation)
    other = object()
    host.registry.get_probe.side_effect = {"cartographer_touch": probe, "cartographer": other}.get
    host.manager.get_z_compensation_value.return_value = correction
    incoming = cast("GCodeCommand", Mock())
    assert probe.get_status(0)["last_z_result"] is None
    assert probe.get_offsets() == (0, 0, 0)
    assert probe.run_probe(incoming, object(), unused=True) == [10, 20, 1.5 + correction]
    host.touch.perform_probe.assert_called_once_with(gcmd=incoming)
    host.registry.get_probe.assert_called_once_with("cartographer_touch")
    host.manager.get_z_compensation_value.assert_called_once_with([10, 20, 0], probe=probe)
    status = probe.get_status(1)
    assert status["name"] == "cartographer_touch"
    assert status["last_z_result"] == 1.5 + correction
    assert probe.last_probe_position == Position(10, 20, 1.5 + correction)
    host.touch.offset = Position(1, 2, 3)
    assert probe.get_offsets() == (1, 2, 3)


@pytest.mark.parametrize("stage", ["touch", "compensation"])
def test_errors_propagate_without_overwriting_touch_state(host: Mock, stage: str) -> None:
    probe = KalicoTouchProbe(host.toolhead, host.touch, host.general, host.compensation)
    host.registry.get_probe.return_value = probe
    host.manager.get_z_compensation_value.return_value = 0.0
    incoming = cast("GCodeCommand", Mock())
    _ = probe.run_probe(incoming)
    previous = probe.get_status(0)
    host.toolhead.get_position.return_value = Position(30, 40, 5)
    host.touch.perform_probe.return_value = 2.5
    failing = host.touch.perform_probe if stage == "touch" else host.manager.get_z_compensation_value
    failing.side_effect = RuntimeError("reference failure")
    with pytest.raises(ValueError, match="reference failure"):
        _ = probe.run_probe(incoming)
    assert probe.get_status(1) == previous
    assert host.manager.get_z_compensation_value.call_count == (1 if stage == "touch" else 2)


@pytest.mark.parametrize("position", [Position(-1, 20, 5), Position(10, 101, 5)])
def test_bounds_checked_before_touch_execution(host: Mock, position: Position) -> None:
    probe = KalicoTouchProbe(host.toolhead, host.touch, host.general, host.compensation)
    host.toolhead.get_position.return_value = position
    with pytest.raises(ValueError, match="outside touch boundaries"):
        _ = probe.run_probe(cast("GCodeCommand", Mock()))
    host.touch.perform_probe.assert_not_called()
    host.manager.get_z_compensation_value.assert_not_called()
    assert probe.get_status(0)["last_z_result"] is None


def test_lift_speed_uses_positive_native_override(host: Mock) -> None:
    probe = KalicoTouchProbe(host.toolhead, host.touch, host.general, host.compensation)
    incoming = Mock()
    incoming.get_float.return_value = 8.0
    assert probe.get_lift_speed() == 6.0
    assert probe.get_lift_speed(incoming) == 8.0
    incoming.get_float.assert_called_once_with("LIFT_SPEED", 6.0, above=0.0)
