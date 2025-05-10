from __future__ import annotations

from typing import TYPE_CHECKING, final

import pytest
from typing_extensions import TypeAlias

from cartographer.interfaces.printer import HomingState, Mcu, Position, Sample, TemperatureStatus, Toolhead
from cartographer.probe.touch_mode import Configuration, TouchMode

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

    from cartographer.interfaces.configuration import TouchModelConfiguration


Probe: TypeAlias = TouchMode


@final
class MockConfiguration(Configuration):
    move_speed = 42.0

    touch_samples = 5
    touch_max_samples = 10
    x_offset = 10.0
    y_offset = 10.0
    mesh_min = (10, 10)
    mesh_max = (100, 100)


class MockModel:
    name: str = "default"
    threshold: int = 5
    speed: float = 10.0
    z_offset: float = 0.0

    def save_z_offset(self, new_offset: float) -> None:
        del new_offset
        pass


@pytest.fixture
def config() -> Configuration:
    return MockConfiguration()


@pytest.fixture
def model() -> TouchModelConfiguration:
    return MockModel()


@pytest.fixture
def probe(mcu: Mcu[object, Sample], toolhead: Toolhead, config: Configuration, model: TouchModelConfiguration) -> Probe:
    return Probe(mcu, toolhead, config, model=model)


@pytest.fixture
def homing_state(mocker: MockerFixture, probe: Probe) -> HomingState:
    mock = mocker.Mock(spec=HomingState, autospec=True)
    mock.endstops = [probe]
    return mock


def test_probe_success(mocker: MockerFixture, toolhead: Toolhead, probe: Probe) -> None:
    toolhead.z_homing_move = mocker.Mock(return_value=0.5)
    toolhead.get_position = mocker.Mock(return_value=Position(0, 0, 1))

    assert probe.perform_probe() == 0.5


def test_probe_moves_below_5(mocker: MockerFixture, toolhead: Toolhead, probe: Probe) -> None:
    toolhead.z_homing_move = mocker.Mock(return_value=0.5)
    toolhead.get_position = mocker.Mock(return_value=Position(0, 0, 1))
    move_spy = mocker.spy(toolhead, "move")

    _ = probe.perform_probe()

    assert move_spy.mock_calls[0] == mocker.call(z=5, speed=mocker.ANY)


def test_does_not_move_above_5(mocker: MockerFixture, toolhead: Toolhead, probe: Probe) -> None:
    toolhead.z_homing_move = mocker.Mock(return_value=0.5)
    toolhead.get_position = mocker.Mock(return_value=Position(0, 0, 10))
    move_spy = mocker.spy(toolhead, "move")

    _ = probe.perform_probe()

    assert move_spy.mock_calls[0] != mocker.call(z=5, speed=mocker.ANY)


def test_probe_standard_deviation_failure(
    mocker: MockerFixture, toolhead: Toolhead, probe: Probe, config: MockConfiguration
) -> None:
    config.touch_max_samples = 5
    toolhead.z_homing_move = mocker.Mock(side_effect=[1.000, 1.002, 1.1, 1.016, 1.018])
    toolhead.get_position = mocker.Mock(return_value=Position(0, 0, 1))

    with pytest.raises(RuntimeError, match="unable to find"):
        _ = probe.perform_probe()


def test_probe_suceeds_on_more(mocker: MockerFixture, toolhead: Toolhead, probe: Probe) -> None:
    toolhead.z_homing_move = mocker.Mock(side_effect=[1.0, 1.01, 1.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    toolhead.get_position = mocker.Mock(return_value=Position(0, 0, 1))

    assert probe.perform_probe() == 0.5


def test_probe_suceeds_on_spread_samples(mocker: MockerFixture, toolhead: Toolhead, probe: Probe) -> None:
    toolhead.z_homing_move = mocker.Mock(side_effect=[0.5, 1.0, 1.5, 0.5, 2.5, 0.5, 3.5, 0.5, 4.5, 0.5])
    toolhead.get_position = mocker.Mock(return_value=Position(0, 0, 1))

    assert probe.perform_probe() == 0.5


def test_probe_unhomed_z(mocker: MockerFixture, toolhead: Toolhead, probe: Probe) -> None:
    toolhead.is_homed = mocker.Mock(return_value=False)

    with pytest.raises(RuntimeError, match="z axis must be homed"):
        _ = probe.perform_probe()


def test_home_start_invalid_threshold(model: TouchModelConfiguration, probe: Probe) -> None:
    model.threshold = 0

    with pytest.raises(RuntimeError, match="threshold must be greater than 0"):
        _ = probe.home_start(print_time=0.0)


def test_home_wait(mocker: MockerFixture, mcu: Mcu[object, Sample], probe: Probe) -> None:
    mcu.stop_homing = mocker.Mock(return_value=1.5)

    assert probe.home_wait(home_end_time=1.0) == 1.5


def test_on_home_end(mocker: MockerFixture, probe: Probe, homing_state: HomingState) -> None:
    homed_position_spy = mocker.spy(homing_state, "set_z_homed_position")

    probe.on_home_end(homing_state)

    assert homed_position_spy.called == 1


def test_abort_if_current_extruder_too_hot(mocker: MockerFixture, toolhead: Toolhead, probe: Probe) -> None:
    toolhead.get_extruder_temperature = mocker.Mock(return_value=TemperatureStatus(156, 0))

    with pytest.raises(RuntimeError, match="nozzle temperature must be below 150C"):
        _ = probe.home_start(print_time=0.0)


def test_abort_if_current_extruder_target_too_hot(mocker: MockerFixture, toolhead: Toolhead, probe: Probe) -> None:
    toolhead.get_extruder_temperature = mocker.Mock(return_value=TemperatureStatus(0, 156))

    with pytest.raises(RuntimeError, match="nozzle temperature must be below 150C"):
        _ = probe.home_start(print_time=0.0)


def test_nozzle_outside_bounds(mocker: MockerFixture, toolhead: Toolhead, probe: Probe) -> None:
    toolhead.get_position = mocker.Mock(return_value=Position(0, 0, 1))

    with pytest.raises(RuntimeError, match="outside .* boundaries"):
        _ = probe.home_start(0)


def test_probe_outside_bounds(mocker: MockerFixture, toolhead: Toolhead, probe: Probe) -> None:
    toolhead.get_position = mocker.Mock(return_value=Position(95, 95, 1))

    with pytest.raises(RuntimeError, match="outside .* boundaries"):
        _ = probe.home_start(0)
