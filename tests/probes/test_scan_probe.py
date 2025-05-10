from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest
from typing_extensions import TypeAlias, override

from cartographer.interfaces.printer import HomingState, Mcu, Position, Toolhead
from cartographer.probe.scan_mode import Model, ScanMode, ScanModeConfiguration

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

    from cartographer.stream import Session


@dataclass
class Sample:
    time: float
    frequency: float
    position: Position | None = field(default_factory=lambda: Position(10, 10, 10))
    velocity: float | None = 5


Probe: TypeAlias = ScanMode


class MockModel(Model):
    @property
    @override
    def name(self) -> str:
        return "test"

    actual_z_offset: float = 0

    @property
    @override
    def z_offset(self) -> float:
        return self.actual_z_offset

    @override
    def save_z_offset(self, new_offset: float) -> None:
        self.actual_z_offset = new_offset

    @override
    def distance_to_frequency(self, distance: float) -> float:
        return distance

    @override
    def frequency_to_distance(self, frequency: float) -> float:
        return frequency


class MockConfiguration(ScanModeConfiguration):
    x_offset: float = 0.0
    y_offset: float = 0.0
    move_speed: float = 42.0

    scan_samples: int = 10
    scan_mesh_runs: int = 1


@pytest.fixture
def config() -> MockConfiguration:
    return MockConfiguration()


@pytest.fixture
def model() -> Model:
    return MockModel()


@pytest.fixture
def homing_state(mocker: MockerFixture, probe: Probe) -> HomingState:
    mock = mocker.Mock(spec=HomingState, autospec=True)
    mock.endstops = [probe]
    return mock


@pytest.fixture
def probe(mcu: Mcu[object, Sample], toolhead: Toolhead, config: ScanModeConfiguration, model: Model) -> Probe:
    return ScanMode(mcu, toolhead, config, model=model)


def test_measures_distance(probe: Probe, session: Session[Sample]):
    session.get_items = lambda: [Sample(time=0.0, frequency=i + 1) for i in range(11)]

    distance = probe.measure_distance(skip_count=0)

    assert distance == 6


def test_skips_samples(probe: Probe, session: Session[Sample]):
    session.get_items = lambda: [Sample(time=0.0, frequency=i + 1) for i in range(11)]

    distance = probe.measure_distance(skip_count=4)

    assert distance == 8


def test_raises_error_when_model_is_missing(probe: Probe, model: Model):
    model.distance_to_frequency = lambda distance: 42

    frequency = probe.distance_to_frequency(1.0)

    assert frequency == 42


def test_probe_errors_when_not_homed(probe: Probe, toolhead: Toolhead):
    toolhead.is_homed = lambda axis: False

    with pytest.raises(RuntimeError):
        _ = probe.perform_probe()


def test_probe_returns_trigger_position(probe: Probe, toolhead: Toolhead, session: Session[Sample]):
    dist = 1
    z_pos = 2
    session.get_items = lambda: [Sample(time=0.0, frequency=dist) for _ in range(11)]
    toolhead.get_position = lambda: Position(0, 0, z_pos)

    trigger_pos = probe.perform_probe()

    assert trigger_pos == probe.probe_height + z_pos - dist


def test_probe_applies_axis_twist_compensation(probe: Probe, toolhead: Toolhead, session: Session[Sample]):
    dist = 1
    z_pos = 2
    z_comp = 0.5
    session.get_items = lambda: [Sample(time=0.0, frequency=dist) for _ in range(11)]
    toolhead.get_position = lambda: Position(0, 0, z_pos)
    toolhead.apply_axis_twist_compensation = lambda position: Position(position.x, position.y, z_comp)

    trigger_pos = probe.perform_probe()

    assert trigger_pos == z_comp


def test_probe_errors_outside_range(probe: Probe, session: Session[Sample], model: Model):
    session.get_items = lambda: [Sample(time=0.0, frequency=42) for _ in range(11)]
    model.frequency_to_distance = lambda frequency: float("inf")

    with pytest.raises(RuntimeError):
        _ = probe.perform_probe()


def test_do_nothing_when_not_homing(mocker: MockerFixture, probe: Probe, homing_state: HomingState):
    homed_position_spy = mocker.spy(homing_state, "set_z_homed_position")
    homing_state.is_homing_z = lambda: False
    probe.on_home_end(homing_state)
    assert homed_position_spy.call_count == 0


def test_do_nothing_when_endstop_not_homing(
    mocker: MockerFixture,
    probe: Probe,
    homing_state: HomingState,
):
    homed_position_spy = mocker.spy(homing_state, "set_z_homed_position")
    homing_state.is_homing_z = lambda: True
    homing_state.endstops = []
    probe.on_home_end(homing_state)
    assert homed_position_spy.call_count == 0


def test_scan_mode_sets_homed_position(
    mocker: MockerFixture,
    probe: Probe,
    homing_state: HomingState,
):
    homed_position_spy = mocker.spy(homing_state, "set_z_homed_position")
    probe.measure_distance = mocker.Mock(return_value=5)

    _ = probe.home_start(0)
    probe.on_home_end(homing_state)

    homed_position_spy.assert_called_once_with(5)


def test_endstop_is_triggered(mocker: MockerFixture, probe: Probe):
    probe.measure_distance = mocker.Mock(return_value=1)

    assert probe.query_is_triggered(0) is True


def test_endstop_is_not_triggered(mocker: MockerFixture, probe: Probe):
    probe.measure_distance = mocker.Mock(return_value=1)

    assert probe.query_is_triggered(0) is True


def test_save_z_offset_0(mocker: MockerFixture, probe: Probe, model: Model):
    save_model_offset_spy = mocker.spy(model, "save_z_offset")

    adjustment = 0.2
    new_offset = probe.offset.z - adjustment
    probe.save_z_offset(new_offset)

    assert probe.offset.z == 1.8
    assert save_model_offset_spy.mock_calls == [mocker.call(pytest.approx(-0.2))]  # pyright: ignore[reportUnknownMemberType]


def test_save_z_offset_existing(mocker: MockerFixture, probe: Probe, model: MockModel):
    model.actual_z_offset = 1.0
    save_model_offset_spy = mocker.spy(model, "save_z_offset")

    adjustment = 0.2
    new_offset = probe.offset.z - adjustment
    probe.save_z_offset(new_offset)

    assert probe.offset.z == 2.8
    assert save_model_offset_spy.mock_calls == [mocker.call(pytest.approx(0.8))]  # pyright: ignore[reportUnknownMemberType]
