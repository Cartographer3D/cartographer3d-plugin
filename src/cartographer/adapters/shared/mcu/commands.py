from __future__ import annotations

import logging
from enum import IntEnum
from typing import TYPE_CHECKING, NamedTuple, final

if TYPE_CHECKING:
    from mcu import MCU, CommandWrapper

logger = logging.getLogger(__name__)


class TriggerMethod(IntEnum):
    SCAN = 0
    TOUCH = 1


class HomeCommand(NamedTuple):
    trsync_oid: int
    trigger_reason: int
    trigger_invert: int
    threshold: int
    trigger_method: TriggerMethod


class ThresholdCommand(NamedTuple):
    trigger: int
    untrigger: int


@final
class KlipperCartographerCommands:
    def __init__(self, mcu: MCU):
        self._command_queue = mcu.alloc_command_queue()
        self._stream_command = mcu.lookup_command("cartographer_stream en=%u", cq=self._command_queue)
        self._set_threshold_command = mcu.lookup_command(
            "cartographer_set_threshold trigger=%u untrigger=%u", cq=self._command_queue
        )
        self._start_home_command = mcu.lookup_command(
            "cartographer_home trsync_oid=%c trigger_reason=%c trigger_invert=%c threshold=%u trigger_method=%u",
            cq=self._command_queue,
        )
        self._stop_home_command = mcu.lookup_command("cartographer_stop_home", cq=self._command_queue)

    def _ensure_initialized(self, command: CommandWrapper | None, name: str) -> CommandWrapper:
        if command is None:
            msg = f"command {name} has not been initialized"
            raise RuntimeError(msg)
        return command

    def send_stream_state(self, *, enable: bool) -> None:
        command = self._ensure_initialized(self._stream_command, "stream command")
        logger.debug("%s stream", "Starting" if enable else "Stopping")
        command.send([1 if enable else 0])

    def send_threshold(self, command: ThresholdCommand) -> None:
        cmd = self._ensure_initialized(self._set_threshold_command, "set threshold command")
        logger.debug("Sending trigger frequency threshold command %s", list(command))
        cmd.send(list(command))

    def send_home(self, command: HomeCommand) -> None:
        cmd = self._ensure_initialized(self._start_home_command, "start home command")
        logger.debug("Sending home command %s", list(command))
        cmd.send(list(command))

    def send_stop_home(self) -> None:
        cmd = self._ensure_initialized(self._stop_home_command, "stop home command")
        logger.debug("Sending stop home command")
        cmd.send()
