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
