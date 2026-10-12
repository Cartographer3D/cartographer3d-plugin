from __future__ import annotations

import sys
from typing import TYPE_CHECKING
from unittest.mock import Mock

# Stub klipper modules not present in test environment
if "mcu" not in sys.modules:
    sys.modules["mcu"] = Mock()

from cartographer.mcu.commands import (
    CartographerCommands,
    HomeCommand,
    ThresholdCommand,
    TriggerMethod,
)

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def _make_commands(mocker: MockerFixture) -> tuple[CartographerCommands, Mock]:
    mcu = mocker.MagicMock()
    cmd_wrapper = mocker.Mock()
    mcu.lookup_command.return_value = cmd_wrapper
    mcu.alloc_command_queue.return_value = mocker.Mock()
    commands = CartographerCommands(mcu)
    commands.initialize()
    return commands, cmd_wrapper


class TestSendCommands:
    """Smoke tests that each public send method dispatches via CommandWrapper.send."""

    def test_send_stream_state_enable(self, mocker: MockerFixture) -> None:
        commands, wrapper = _make_commands(mocker)
        commands.send_stream_state(enable=True)
        wrapper.send.assert_called_once_with([1])

    def test_send_stream_state_disable(self, mocker: MockerFixture) -> None:
        commands, wrapper = _make_commands(mocker)
        commands.send_stream_state(enable=False)
        wrapper.send.assert_called_once_with([0])

    def test_send_threshold(self, mocker: MockerFixture) -> None:
        commands, wrapper = _make_commands(mocker)
        commands.send_threshold(ThresholdCommand(trigger=100, untrigger=90))
        wrapper.send.assert_called_once_with([100, 90])

    def test_send_home(self, mocker: MockerFixture) -> None:
        commands, wrapper = _make_commands(mocker)
        home_cmd = HomeCommand(
            trsync_oid=1,
            trigger_reason=2,
            trigger_invert=0,
            threshold=50,
            trigger_method=TriggerMethod.SCAN,
        )
        commands.send_home(home_cmd)
        wrapper.send.assert_called_once_with(list(home_cmd))

    def test_send_stop_home(self, mocker: MockerFixture) -> None:
        commands, wrapper = _make_commands(mocker)
        commands.send_stop_home()
        wrapper.send.assert_called_once_with()


def _make_commands_with_query(mocker: MockerFixture, reply: dict[str, int]) -> CartographerCommands:
    mcu = mocker.MagicMock()
    mcu.lookup_query_command.return_value.send.return_value = reply
    commands = CartographerCommands(mcu)
    commands.initialize()
    return commands


class TestQueryTriggerClock:
    def test_returns_clock_when_triggered(self, mocker: MockerFixture) -> None:
        commands = _make_commands_with_query(mocker, {"triggered": 1, "trigger_clock": 1234})
        assert commands.query_trigger_clock() == 1234

    def test_returns_none_when_not_triggered(self, mocker: MockerFixture) -> None:
        commands = _make_commands_with_query(mocker, {"triggered": 0, "trigger_clock": 1234})
        assert commands.query_trigger_clock() is None

    def test_returns_none_on_firmware_without_query(self, mocker: MockerFixture) -> None:
        mcu = mocker.MagicMock()
        mcu.lookup_query_command.side_effect = Exception("Unknown command: cartographer_query_home")
        commands = CartographerCommands(mcu)
        commands.initialize()
        assert commands.query_trigger_clock() is None
