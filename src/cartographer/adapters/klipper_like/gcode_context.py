from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from gcode import GCodeCommand

_active_command: ContextVar[GCodeCommand | None] = ContextVar("cartographer_gcode_command", default=None)


def get_gcode_command() -> GCodeCommand | None:
    return _active_command.get()


@contextmanager
def gcode_command_context(gcmd: GCodeCommand | None) -> Iterator[None]:
    token = _active_command.set(gcmd)
    try:
        yield
    finally:
        _active_command.reset(token)
