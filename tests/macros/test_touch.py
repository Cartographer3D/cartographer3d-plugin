from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest
from typing_extensions import TypeAlias

from cartographer.interfaces.printer import MacroParams, Position, Toolhead
from cartographer.macros.touch import TouchAccuracyMacro, TouchHomeMacro, TouchProbeMacro
from cartographer.probe.touch_mode import TouchMode, TouchModeConfiguration

if TYPE_CHECKING:
    from pytest import LogCaptureFixture
    from pytest_mock import MockerFixture


Probe: TypeAlias = TouchMode


@pytest.fixture
def offset() -> Position:
    return Position(0, 0, 0)


@pytest.fixture
def probe(mocker: MockerFixture, offset: Position) -> Probe:
    mock = mocker.Mock(spec=Probe, autospec=True)
    mock.config = mocker.Mock(spec=TouchModeConfiguration, autospec=True)
    mock.config.move_speed = 42
    mock.offset = offset
    return mock


def test_touch_macro_output(
    mocker: MockerFixture,
    caplog: LogCaptureFixture,
    probe: Probe,
    params: MacroParams,
):
    macro = TouchProbeMacro(probe)
    probe.perform_probe = mocker.Mock(return_value=5.0)

    with caplog.at_level(logging.INFO):
        macro.run(params)

    assert "Result is z=5.000000" in caplog.messages


def test_touch_accuracy_macro_output(
    mocker: MockerFixture,
    caplog: LogCaptureFixture,
    probe: Probe,
    toolhead: Toolhead,
    params: MacroParams,
):
    macro = TouchAccuracyMacro(probe, toolhead, lift_speed=5)
    params.get_int = mocker.Mock(return_value=10)
    toolhead.get_position = lambda: Position(0, 0, 0)
    params.get_float = mocker.Mock(return_value=1)
    i = -1
    measurements: list[float] = [50 + i * 10 for i in range(10)]

    def mock_probe(**_) -> float:
        nonlocal i
        i += 1
        return measurements[i]

    probe.perform_probe = mock_probe

    with caplog.at_level(logging.INFO):
        macro.run(params)

    assert "touch accuracy results" in caplog.text
    assert "minimum 50" in caplog.text
    assert "maximum 140" in caplog.text
    assert "range 90" in caplog.text
    assert "average 95" in caplog.text
    assert "median 95" in caplog.text
    assert "standard deviation 28" in caplog.text


def test_touch_accuracy_macro_sample_count(
    mocker: MockerFixture,
    caplog: LogCaptureFixture,
    probe: Probe,
    toolhead: Toolhead,
    params: MacroParams,
):
    macro = TouchAccuracyMacro(probe, toolhead, lift_speed=5)
    params.get_int = mocker.Mock(return_value=3)
    toolhead.get_position = lambda: Position(0, 0, 0)
    params.get_float = mocker.Mock(return_value=1)
    i = -1
    measurements: list[float] = [50 + i * 10 for i in range(10)]

    def mock_probe(**_) -> float:
        nonlocal i
        i += 1
        return measurements[i]

    probe.perform_probe = mock_probe

    with caplog.at_level(logging.INFO):
        macro.run(params)

    assert "touch accuracy results" in caplog.text
    assert "minimum 50" in caplog.text
    assert "maximum 70" in caplog.text
    assert "range 20" in caplog.text
    assert "average 60" in caplog.text
    assert "median 60" in caplog.text
    assert "standard deviation 8" in caplog.text


def test_touch_home_macro_moves(
    mocker: MockerFixture,
    probe: Probe,
    toolhead: Toolhead,
    params: MacroParams,
):
    macro = TouchHomeMacro(probe, toolhead, home_position=(10, 10), lift_speed=5, travel_speed=50, random_radius=0)
    probe.perform_probe = mocker.Mock(return_value=0.1)
    toolhead.get_position = mocker.Mock(return_value=Position(0, 0, 2))
    move_spy = mocker.spy(toolhead, "move")

    macro.run(params)

    assert move_spy.mock_calls == [
        mocker.call(z=4, speed=mocker.ANY),
        mocker.call(x=10, y=10, speed=mocker.ANY),
    ]


def test_touch_home_macro(
    mocker: MockerFixture,
    probe: Probe,
    toolhead: Toolhead,
    params: MacroParams,
):
    # We are at 2, and touch the bed at -0.1.
    height = 2
    trigger = 0.1
    # That means that the bed was further away than we thought,
    # so we need to move the z axis "down".
    expected = height - trigger

    macro = TouchHomeMacro(probe, toolhead, home_position=(10, 10), lift_speed=5, travel_speed=50, random_radius=0)
    probe.perform_probe = mocker.Mock(return_value=trigger)
    toolhead.get_position = mocker.Mock(return_value=Position(0, 0, height))
    set_z_position_spy = mocker.spy(toolhead, "set_z_position")

    macro.run(params)

    assert set_z_position_spy.mock_calls == [mocker.call(expected)]


def test_unhomed_touch_home_macro(
    mocker: MockerFixture,
    probe: Probe,
    toolhead: Toolhead,
    params: MacroParams,
):
    toolhead.is_homed = lambda axis: axis != "z"
    max_offset = 10
    height = toolhead.get_axis_limits("z")[1] - max_offset
    trigger = 0.1
    # That means that the bed was further away than we thought,
    # so we need to move the z axis "down".
    expected = height - trigger

    macro = TouchHomeMacro(probe, toolhead, home_position=(10, 10), lift_speed=5, travel_speed=50, random_radius=0)
    probe.perform_probe = mocker.Mock(return_value=trigger)
    toolhead.get_position = mocker.Mock(return_value=Position(0, 0, height))
    set_z_position_spy = mocker.spy(toolhead, "set_z_position")

    macro.run(params)

    assert set_z_position_spy.mock_calls == [mocker.call(z=height), mocker.call(expected)]


@pytest.mark.parametrize(
    "u1, u2, expected_x, expected_y",
    [
        (0.0, 0.0, 50.0, 50.0),  # Center point
        (1.0, 0.0, 60.0, 50.0),  # Max radius, 0 angle
        (0.25, 0.25, 50.0, 55.0),  # radius=5, angle=π/2
        (0.16, 0.5, 46.0, 50.0),  # radius=4, angle=π
    ],
)
def test_random_radius_uniform_distribution(
    mocker: MockerFixture,
    probe: TouchMode,
    toolhead: Toolhead,
    params: MacroParams,
    u1: float,
    u2: float,
    expected_x: float,
    expected_y: float,
):
    """Test that random positions are generated correctly with square root method."""
    macro = TouchHomeMacro(probe, toolhead, home_position=(50, 50), lift_speed=5, travel_speed=50, random_radius=10.0)
    probe.perform_probe = mocker.Mock(return_value=0.1)
    toolhead.get_position = mocker.Mock(return_value=Position(0, 0, 2))
    move_spy = mocker.spy(toolhead, "move")

    _ = mocker.patch("cartographer.macros.touch.home.random", side_effect=[u1, u2])
    macro.run(params)

    assert move_spy.mock_calls == [
        mocker.call(z=4, speed=mocker.ANY),
        mocker.call(x=expected_x, y=expected_y, speed=mocker.ANY),
    ]
