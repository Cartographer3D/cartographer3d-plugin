from __future__ import annotations

from inspect import Parameter, signature
from typing import Callable

from typing_extensions import Protocol


class _Toolhead(Protocol):
    @property
    def set_position(self) -> Callable[..., None]: ...


def uses_string_homing_axes(toolhead: _Toolhead) -> bool:
    """Return whether ``ToolHead.set_position`` expects string homing axes."""
    try:
        homing_axes = signature(toolhead.set_position).parameters["homing_axes"]
    except (TypeError, ValueError, KeyError):
        return True

    return homing_axes.default is Parameter.empty or isinstance(homing_axes.default, str)
