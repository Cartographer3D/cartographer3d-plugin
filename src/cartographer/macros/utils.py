from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterable, Iterator, TypeVar

if TYPE_CHECKING:
    from cartographer.interfaces.printer import MacroParams, Toolhead

K = TypeVar("K", bound=str)


def get_choice(params: MacroParams, option: str, choices: Iterable[K], default: K) -> K:
    choice = params.get(option, default=default)
    choice_str = choice.lower()

    for k in choices:
        if k.lower() == choice_str:
            return k

    valid_choices = ", ".join(f"'{k.lower()}'" for k in choices)
    msg = f"Invalid choice '{choice}' for option '{option}'. Valid choices are: {valid_choices}"
    raise RuntimeError(msg)


def get_int_tuple(params: MacroParams, option: str, default: tuple[int, int]) -> tuple[int, int]:
    param = params.get(option, default=None)
    if param is None:
        return default
    parts = param.split(",")
    if len(parts) != 2:
        msg = f"Expected two int values for '{option}', got {len(parts)}: {param}"
        raise ValueError(msg)

    return (int(parts[0]), int(parts[1]))


def get_float_tuple(params: MacroParams, option: str, default: tuple[float, float]) -> tuple[float, float]:
    param = params.get(option, default=None)
    if param is None:
        return default
    parts = param.split(",")
    if len(parts) != 2:
        msg = f"Expected two float values for '{option}', got {len(parts)}: {param}"
        raise ValueError(msg)

    return (float(parts[0]), float(parts[1]))


@contextmanager
def force_home_z(toolhead: Toolhead, offset: float = 10) -> Iterator[None]:
    """
    Context manager that temporarily sets a forced Z position for homing operations.

    If the Z axis is already homed, this context manager does nothing.
    If the Z axis is not homed, it temporarily sets a forced Z position
    at `z_max - offset` and clears the homing state on exit.

    Parameters
    ----------
    toolhead : Toolhead
        The toolhead instance to manage Z positioning for.
    offset : float, optional
        Distance below Z maximum to set as temporary position, by default 10.
    """
    if toolhead.is_homed("z"):
        yield
        return

    _, z_max = toolhead.get_axis_limits("z")
    toolhead.set_z_position(z=z_max - offset)

    try:
        yield
    finally:
        toolhead.clear_z_homing_state()
