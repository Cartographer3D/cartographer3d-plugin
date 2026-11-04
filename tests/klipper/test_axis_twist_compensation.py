from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NamedTuple
from unittest.mock import Mock

import pytest
from extras import manual_probe

from cartographer.adapters.klipper.axis_twist_compensation import KlipperAxisTwistCompensationAdapter

if TYPE_CHECKING:
    from pytest import MonkeyPatch


class ProbeResult(NamedTuple):
    bed_x: float
    bed_y: float
    bed_z: float
    test_x: float
    test_y: float
    test_z: float


@pytest.fixture
def firmware_config() -> Mock:
    printer = Mock()
    config = Mock()
    config.get_printer.return_value = printer
    config.getsection.return_value = Mock()
    printer.load_object.return_value = Mock(horizontal_move_z=5.0, speed=50.0)
    return config


def test_legacy_list_preserves_toolhead_coordinates(firmware_config: Mock) -> None:
    assert not hasattr(manual_probe, "ProbeResult")

    def update_results(event: str, position: list[float]) -> None:
        assert event == "probe:update_results"
        assert position == [41.0, 67.0, 0.0]
        position[2] = 0.375

    firmware_config.get_printer.return_value.send_event.side_effect = update_results
    adapter = KlipperAxisTwistCompensationAdapter(firmware_config, x_offset=12.5, y_offset=-23.0)

    assert adapter.get_z_compensation_value(x=41.0, y=67.0) == 0.375


@pytest.mark.parametrize("frame", ["test", "bed"])
@pytest.mark.parametrize("x_offset, y_offset", [(12.5, -23.0), (-8.0, 14.5)])
def test_probe_result_preserves_both_coordinate_frames(
    monkeypatch: MonkeyPatch,
    firmware_config: Mock,
    frame: Literal["test", "bed"],
    x_offset: float,
    y_offset: float,
) -> None:
    monkeypatch.setattr(manual_probe, "ProbeResult", ProbeResult, raising=False)

    def update_results(event: str, results: list[ProbeResult]) -> None:
        assert event == "probe:update_results"
        assert results == [ProbeResult(41.0 + x_offset, 67.0 + y_offset, 0.0, 41.0, 67.0, 0.0)]
        result = results[0]
        if frame == "test":
            x, y = result.test_x, result.test_y
        else:
            x, y = result.bed_x, result.bed_y
        results[0] = result._replace(bed_z=x * 0.01 + y * 0.02, test_z=-99.0)

    firmware_config.get_printer.return_value.send_event.side_effect = update_results
    adapter = KlipperAxisTwistCompensationAdapter(firmware_config, x_offset=x_offset, y_offset=y_offset)
    expected_x = 41.0 if frame == "test" else 41.0 + x_offset
    expected_y = 67.0 if frame == "test" else 67.0 + y_offset

    assert adapter.get_z_compensation_value(x=41.0, y=67.0) == expected_x * 0.01 + expected_y * 0.02
