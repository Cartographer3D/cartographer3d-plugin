from __future__ import annotations

import sys
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

# Stub klipper modules not present in test environment
if "mcu" not in sys.modules:
    sys.modules["mcu"] = Mock()

from cartographer.adapters.klipper.mcu.commands import (
    HomeCommand,
    KlipperCartographerCommands,
    ThresholdCommand,
    TriggerMethod,
)

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


class TestCheckConnected:
    """Tests for the _check_connected pre-flight guard."""

    def _make_commands(self, mocker: MockerFixture, *, disconnected: bool) -> tuple[KlipperCartographerCommands, Mock]:
        mcu = mocker.MagicMock()
        mcu.non_critical_disconnected = disconnected
        cmd_wrapper = mocker.Mock()
        mcu.lookup_command.return_value = cmd_wrapper
        mcu.alloc_command_queue.return_value = mocker.Mock()
        commands = KlipperCartographerCommands(mcu)
        commands.initialize()
        return commands, cmd_wrapper

    # --- send_stream_state ---

    def test_send_stream_state_normal(self, mocker: MockerFixture) -> None:
        commands, wrapper = self._make_commands(mocker, disconnected=False)
        commands.send_stream_state(enable=True)
        wrapper.send.assert_called_once_with([1])

    def test_send_stream_state_disconnected_raises_runtime_error(self, mocker: MockerFixture) -> None:
        commands, wrapper = self._make_commands(mocker, disconnected=True)
        with pytest.raises(RuntimeError, match="disconnected"):
            commands.send_stream_state(enable=True)
        wrapper.send.assert_not_called()

    # --- send_threshold ---

    def test_send_threshold_normal(self, mocker: MockerFixture) -> None:
        commands, wrapper = self._make_commands(mocker, disconnected=False)
        threshold = ThresholdCommand(trigger=100, untrigger=90)
        commands.send_threshold(threshold)
        wrapper.send.assert_called_once_with([100, 90])

    def test_send_threshold_disconnected_raises_runtime_error(self, mocker: MockerFixture) -> None:
        commands, wrapper = self._make_commands(mocker, disconnected=True)
        with pytest.raises(RuntimeError, match="disconnected"):
            commands.send_threshold(ThresholdCommand(trigger=100, untrigger=90))
        wrapper.send.assert_not_called()

    # --- send_home ---

    def test_send_home_normal(self, mocker: MockerFixture) -> None:
        commands, wrapper = self._make_commands(mocker, disconnected=False)
        home_cmd = HomeCommand(
            trsync_oid=1,
            trigger_reason=2,
            trigger_invert=0,
            threshold=50,
            trigger_method=TriggerMethod.SCAN,
        )
        commands.send_home(home_cmd)
        wrapper.send.assert_called_once_with(list(home_cmd))

    def test_send_home_disconnected_raises_runtime_error(self, mocker: MockerFixture) -> None:
        commands, wrapper = self._make_commands(mocker, disconnected=True)
        home_cmd = HomeCommand(
            trsync_oid=1,
            trigger_reason=2,
            trigger_invert=0,
            threshold=50,
            trigger_method=TriggerMethod.SCAN,
        )
        with pytest.raises(RuntimeError, match="disconnected"):
            commands.send_home(home_cmd)
        wrapper.send.assert_not_called()

    # --- send_stop_home ---

    def test_send_stop_home_normal(self, mocker: MockerFixture) -> None:
        commands, wrapper = self._make_commands(mocker, disconnected=False)
        commands.send_stop_home()
        wrapper.send.assert_called_once_with()

    def test_send_stop_home_disconnected_raises_runtime_error(self, mocker: MockerFixture) -> None:
        commands, wrapper = self._make_commands(mocker, disconnected=True)
        with pytest.raises(RuntimeError, match="disconnected"):
            commands.send_stop_home()
        wrapper.send.assert_not_called()

    # --- missing attribute (stock Klipper) ---

    def test_missing_attribute_treated_as_connected(self, mocker: MockerFixture) -> None:
        """On stock Klipper, non_critical_disconnected is absent; getattr default is False."""
        mcu = mocker.MagicMock(
            spec=[
                "alloc_command_queue",
                "lookup_command",
            ]
        )
        cmd_wrapper = mocker.Mock()
        mcu.lookup_command.return_value = cmd_wrapper
        mcu.alloc_command_queue.return_value = mocker.Mock()

        commands = KlipperCartographerCommands(mcu)
        commands.initialize()
        commands.send_stream_state(enable=False)
        cmd_wrapper.send.assert_called_once_with([0])
