from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from cartographer.interfaces.configuration import Configuration, TouchModelConfiguration
from cartographer.interfaces.printer import Mcu, Position, TemperatureStatus, Toolhead

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

    from cartographer.probe.probe import Probe


@pytest.fixture(autouse=True)
def configure_probe(probe: Probe, config: Configuration) -> None:
    config.save_touch_model(
        TouchModelConfiguration(
            name="test_touch",
            speed=3,
            threshold=1000,
            z_offset=0,
            samples=5,
            sample_range=0.010,
        )
    )
    probe.touch.load_model("test_touch")


def test_probe_success(mocker: MockerFixture, toolhead: Toolhead, probe: Probe) -> None:
    toolhead.z_probing_move = mocker.Mock(return_value=0.5)
    toolhead.get_position = mocker.Mock(return_value=Position(0, 0, 1))

    assert probe.touch.perform_probe() == 0.5


def test_probe_includes_z_offset(
    mocker: MockerFixture, toolhead: Toolhead, config: Configuration, probe: Probe
) -> None:
    config.save_touch_model(
        TouchModelConfiguration(
            name="test_touch",
            speed=3,
            threshold=1000,
            z_offset=-0.5,
            samples=5,
            sample_range=0.010,
        )
    )
    probe.touch.load_model("test_touch")
    toolhead.get_axis_limits = mocker.Mock(return_value=(-5, 100))
    toolhead.z_probing_move = mocker.Mock(return_value=-0.5)
    toolhead.get_position = mocker.Mock(return_value=Position(0, 0, 1))

    assert probe.touch.perform_probe() == 0


def test_probe_moves_below_2(mocker: MockerFixture, toolhead: Toolhead, probe: Probe) -> None:
    toolhead.z_probing_move = mocker.Mock(return_value=0.5)
    toolhead.get_position = mocker.Mock(return_value=Position(0, 0, 1))
    move_spy = mocker.spy(toolhead, "move")

    _ = probe.touch.perform_probe()

    assert move_spy.mock_calls[0] == mocker.call(z=2, speed=mocker.ANY)


def test_does_not_move_above_2(mocker: MockerFixture, toolhead: Toolhead, probe: Probe) -> None:
    toolhead.z_probing_move = mocker.Mock(return_value=0.5)
    toolhead.get_position = mocker.Mock(return_value=Position(0, 0, 10))
    move_spy = mocker.spy(toolhead, "move")

    _ = probe.touch.perform_probe()

    assert move_spy.mock_calls[0] != mocker.call(z=2, speed=mocker.ANY)


def test_probe_standard_deviation_failure(mocker: MockerFixture, toolhead: Toolhead, probe: Probe) -> None:
    toolhead.z_probing_move = mocker.Mock(side_effect=[1 + i * 0.1 for i in range(20)])
    toolhead.get_position = mocker.Mock(return_value=Position(0, 0, 1))

    with pytest.raises(RuntimeError, match="Unable to find"):
        _ = probe.touch.perform_probe()


def test_probe_suceeds_on_more(mocker: MockerFixture, toolhead: Toolhead, probe: Probe) -> None:
    toolhead.z_probing_move = mocker.Mock(side_effect=[1.0, 1.01, 1.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    toolhead.get_position = mocker.Mock(return_value=Position(0, 0, 1))

    assert probe.touch.perform_probe() == 0.5


def test_probe_spread_samples_rejected_by_window(mocker: MockerFixture, toolhead: Toolhead, probe: Probe) -> None:
    # Spread-out good samples interleaved with bad ones can no longer be cherry-picked
    # because the sliding window only considers the most recent samples + max_noisy_samples.
    toolhead.z_probing_move = mocker.Mock(side_effect=[0.5, 1.0, 1.5, 0.5, 2.5, 0.5, 3.5, 0.5, 4.5, 0.5])
    toolhead.get_position = mocker.Mock(return_value=Position(0, 0, 1))

    with pytest.raises(RuntimeError, match="Unable to find"):
        _ = probe.touch.perform_probe()


def test_probe_succeeds_within_window(mocker: MockerFixture, toolhead: Toolhead, probe: Probe) -> None:
    # First 3 are noisy, then 5 consistent samples within the window.
    # samples=5, max_noisy_samples=2: window=7, so the last 5 samples [0.5, 0.5, 0.5, 0.5, 0.5] all agree.
    toolhead.z_probing_move = mocker.Mock(side_effect=[1.0, 2.0, 3.0, 0.5, 0.5, 0.5, 0.5, 0.5])
    toolhead.get_position = mocker.Mock(return_value=Position(0, 0, 1))

    assert probe.touch.perform_probe() == 0.5


def test_probe_unhomed_z(mocker: MockerFixture, toolhead: Toolhead, probe: Probe) -> None:
    toolhead.is_homed = mocker.Mock(return_value=False)

    with pytest.raises(RuntimeError, match="Z axis must be homed"):
        _ = probe.touch.perform_probe()


@pytest.mark.parametrize("trigger_z", [-5.0, -4.999], ids=["exact_floor", "inclusive_boundary"])
def test_floor_abort_retract_then_wait_raises(
    mocker: MockerFixture, toolhead: Toolhead, probe: Probe, trigger_z: float
) -> None:
    """Retract queued after floor hit, wait_moves completes it, then RuntimeError raised."""
    toolhead.get_axis_limits = mocker.Mock(return_value=(-5, 100))
    toolhead.z_probing_move = mocker.Mock(return_value=trigger_z)
    toolhead.get_position = mocker.Mock(
        side_effect=[
            Position(0, 0, 2),
            Position(0, 0, 2),
            Position(0, 0, trigger_z),
        ]
    )
    move_counts: list[int] = []
    move_spy = mocker.spy(toolhead, "move")
    wait_spy = mocker.spy(toolhead, "wait_moves")
    wait_spy.side_effect = lambda: move_counts.append(move_spy.call_count)

    with pytest.raises(RuntimeError, match="movement floor"):
        _ = probe.touch.perform_probe()

    move_spy.assert_called_once_with(z=2, speed=5)
    assert move_counts == [0, 0, 1]


def test_home_wait(mocker: MockerFixture, mcu: Mcu, probe: Probe) -> None:
    mcu.stop_homing = mocker.Mock(return_value=1.5)

    assert probe.touch.home_wait(home_end_time=1.0) == 1.5


def test_note_homing_complete_updates_last_homing_time(probe: Probe, toolhead: Toolhead) -> None:
    assert probe.touch.last_homing_time == 0

    probe.touch.note_homing_complete()

    assert probe.touch.last_homing_time == toolhead.get_last_move_time() - 1


def test_abort_if_current_extruder_too_hot(mocker: MockerFixture, toolhead: Toolhead, probe: Probe) -> None:
    toolhead.get_extruder_temperature = mocker.Mock(return_value=TemperatureStatus(156, 0))

    with pytest.raises(RuntimeError, match="Nozzle temperature must be below 150C"):
        _ = probe.touch.home_start(print_time=0.0)


def test_abort_if_current_extruder_target_too_hot(mocker: MockerFixture, toolhead: Toolhead, probe: Probe) -> None:
    toolhead.get_extruder_temperature = mocker.Mock(return_value=TemperatureStatus(0, 156))

    with pytest.raises(RuntimeError, match="Nozzle temperature must be below 150C"):
        _ = probe.touch.home_start(print_time=0.0)


def test_nozzle_outside_bounds(mocker: MockerFixture, toolhead: Toolhead, probe: Probe) -> None:
    toolhead.get_position = mocker.Mock(return_value=Position(-10, 0, 1))

    with pytest.raises(RuntimeError, match="outside .* boundaries"):
        _ = probe.touch.home_start(0)


def test_probe_outside_bounds(mocker: MockerFixture, toolhead: Toolhead, probe: Probe) -> None:
    toolhead.get_position = mocker.Mock(return_value=Position(295, 95, 1))

    with pytest.raises(RuntimeError, match="outside .* boundaries"):
        _ = probe.touch.home_start(0)


def test_increased_budget_permits_longer_sequence(mocker: MockerFixture, toolhead: Toolhead, probe: Probe) -> None:
    """An explicit max_samples override allows convergence that the config budget would exhaust.

    With samples=5, max_noisy_samples=2 (window=7), and sample_range=0.010:
    - 10 diverging values [1.0..1.9] prevent convergence within max_samples=10.
    - Adding 5 more identical values (0.5) brings a clean 5-sample window at attempt 15,
      which succeeds when the budget is raised to 15.
    """
    diverging = [round(1.0 + i * 0.1, 1) for i in range(10)]
    converging = [0.5] * 5
    side_effects = diverging + converging

    toolhead.z_probing_move = mocker.Mock(side_effect=side_effects)
    toolhead.get_position = mocker.Mock(return_value=Position(0, 0, 5))

    # Default budget (max_samples=10) exhausts before convergence.
    with pytest.raises(RuntimeError, match="Unable to find"):
        _ = probe.touch.perform_probe()

    # Reset the mock for the second run.
    toolhead.z_probing_move = mocker.Mock(side_effect=side_effects)

    # Raising the budget to 15 reaches the clean 5-sample window and succeeds.
    result = probe.touch.perform_probe(max_samples=15)
    assert result == 0.5


def test_impossible_budget_fails_before_z_probing_move(mocker: MockerFixture, toolhead: Toolhead, probe: Probe) -> None:
    """max_samples < effective_samples raises RuntimeError before any probing movement."""
    toolhead.z_probing_move = mocker.Mock(return_value=0.5)
    toolhead.get_position = mocker.Mock(return_value=Position(0, 0, 1))
    move_spy = mocker.spy(toolhead, "move")

    # effective samples = 5 (from config); passing max_samples=3 is impossible.
    with pytest.raises(RuntimeError, match=r"max_samples \(3\) must be >= the effective samples \(5\)"):
        _ = probe.touch.perform_probe(max_samples=3)

    move_spy.assert_not_called()
    toolhead.z_probing_move.assert_not_called()
