from __future__ import annotations

import logging
from enum import IntEnum
from typing import TYPE_CHECKING, NamedTuple, final

if TYPE_CHECKING:
    from mcu import CommandQueryWrapper, CommandWrapper

    from cartographer.interfaces.mcu_platform import McuPlatform

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
class CartographerCommands:
    def __init__(self, platform: McuPlatform):
        self._platform = platform
        self._command_queue = platform.alloc_command_queue()
        self._stream_command: CommandWrapper | None = None
        self._set_threshold_command: CommandWrapper | None = None
        self._start_home_command: CommandWrapper | None = None
        self._stop_home_command: CommandWrapper | None = None
        self._query_home_command: CommandQueryWrapper | None = None

    def initialize(self) -> None:
        cq = self._command_queue
        self._stream_command = self._platform.lookup_command("cartographer_stream en=%u", cq=cq)
        self._set_threshold_command = self._platform.lookup_command(
            "cartographer_set_threshold trigger=%u untrigger=%u", cq=cq
        )
        self._start_home_command = self._platform.lookup_command(
            "cartographer_home trsync_oid=%c trigger_reason=%c trigger_invert=%c threshold=%u trigger_method=%u",
            cq=cq,
        )
        self._stop_home_command = self._platform.lookup_command("cartographer_stop_home", cq=cq)
        try:
            self._query_home_command = self._platform.lookup_query_command(
                "cartographer_query_home", "cartographer_home_state triggered=%c trigger_clock=%u", cq=cq
            )
        except Exception:  # firmware without trigger-clock reporting
            self._query_home_command = None

    def _ensure_initialized(self, command: CommandWrapper | None, name: str) -> CommandWrapper:
        if command is None:
            msg = f"Command {name} has not been initialized"
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

    def query_trigger_clock(self) -> int | None:
        """MCU clock (32-bit) of the sample that fired the last home, if the firmware reports it."""
        if self._query_home_command is None:
            return None
        params = self._query_home_command.send()
        return params["trigger_clock"] if params["triggered"] else None
