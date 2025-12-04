from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest
from numpy.polynomial import Polynomial

from cartographer.interfaces.configuration import Configuration, ScanModelConfiguration
from cartographer.interfaces.printer import HomingState, Position, Sample, Toolhead

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

    from cartographer.macros.axis_twist_compensation import AxisTwistCompensationAdapter
    from cartographer.probe.probe import Probe
    from cartographer.stream import Session


@pytest.fixture(autouse=True)
def configure_probe(probe: Probe, config: Configuration) -> None:
    poly = Polynomial([0, 1])
    poly = cast("Polynomial", poly.convert(domain=[0, 20]))

    model = ScanModelConfiguration(
        "test_scan",
        poly.coef,
        poly.domain,
        z_offset=0,
        reference_temperature=40,
    )
    config.save_scan_model(model)
    probe.scan.load_model(model.name)


def sample(
    frequency: float = 1,
    time: float = 0,
    position: Position | None = None,
    temperature: float = 0,
):
    return Sample(
        raw_count=0,
        frequency=1 / frequency,
        time=time,
        position=position,
        temperature=temperature,
    )


def test_measures_distance(probe: Probe, session: Session[Sample]):
    session.get_items = lambda: [sample(frequency=i + 1) for i in range(11)]

    distance = probe.scan.measure_distance(skip_count=0)

    assert distance == pytest.approx(6)  # pyright:ignore[reportUnknownMemberType]


def test_skips_samples(probe: Probe, session: Session[Sample]):
    session.get_items = lambda: [sample(frequency=i + 1) for i in range(11)]

    distance = probe.scan.measure_distance(skip_count=4)

    assert distance == pytest.approx(8)  # pyright:ignore[reportUnknownMemberType]


def test_probe_errors_when_not_homed(probe: Probe, toolhead: Toolhead):
    toolhead.is_homed = lambda axis: False

    with pytest.raises(RuntimeError):
        _ = probe.scan.perform_probe()


def test_probe_returns_trigger_position(probe: Probe, toolhead: Toolhead, session: Session[Sample]):
    dist = 1
    z_pos = 2
    session.get_items = lambda: [sample(frequency=dist) for _ in range(11)]
    toolhead.get_position = lambda: Position(0, 0, z_pos)

    trigger_pos = probe.scan.perform_probe()

    assert trigger_pos == pytest.approx(probe.scan.probe_height + z_pos - dist)  # pyright:ignore[reportUnknownMemberType]


def test_probe_applies_axis_twist_compensation(
    probe: Probe,
    toolhead: Toolhead,
    session: Session[Sample],
    axis_twist_compensation_adapter: AxisTwistCompensationAdapter,
):
    dist = 2
    z_pos = 2
    z_comp = 0.5
    session.get_items = lambda: [sample(frequency=dist) for _ in range(11)]
    toolhead.get_position = lambda: Position(0, 0, z_pos)
    axis_twist_compensation_adapter.get_z_compensation_value = lambda x, y: z_comp

    trigger_pos = probe.scan.perform_probe()

    assert trigger_pos == dist + z_comp


def test_probe_errors_outside_range(probe: Probe, session: Session[Sample]):
    session.get_items = lambda: [sample(frequency=-1) for _ in range(11)]

    with pytest.raises(RuntimeError):
        _ = probe.scan.perform_probe()


def test_do_nothing_when_not_homing(mocker: MockerFixture, probe: Probe, homing_state: HomingState):
    homed_position_spy = mocker.spy(homing_state, "set_z_homed_position")
    homing_state.is_homing_z = lambda: False
    probe.scan.on_home_end(homing_state)
    assert homed_position_spy.call_count == 0


def test_scan_mode_sets_homed_position(
    mocker: MockerFixture,
    probe: Probe,
    homing_state: HomingState,
):
    homed_position_spy = mocker.spy(homing_state, "set_z_homed_position")
    probe.scan.measure_distance = mocker.Mock(return_value=5)

    _ = probe.scan.home_start(0)
    probe.scan.on_home_end(homing_state)

    homed_position_spy.assert_called_once_with(5)


def test_endstop_is_triggered(mocker: MockerFixture, probe: Probe):
    probe.scan.measure_distance = mocker.Mock(return_value=1)

    assert probe.scan.query_is_triggered(0) is True


def test_endstop_is_not_triggered(mocker: MockerFixture, probe: Probe):
    probe.scan.measure_distance = mocker.Mock(return_value=1)

    assert probe.scan.query_is_triggered(0) is True


def test_probe_does_homing_move(mocker: MockerFixture, probe: Probe, toolhead: Toolhead):
    probe.scan.measure_distance = mocker.Mock(return_value=3)
    toolhead.z_probing_move = mocker.Mock(return_value=2)

    _ = probe.scan.perform_probe()

    assert toolhead.z_probing_move.mock_calls == [mocker.call(probe.scan, speed=mocker.ANY)]
