from __future__ import annotations

from contextlib import contextmanager
from enum import Enum
from typing import TYPE_CHECKING, Iterable, Iterator, TypeVar

if TYPE_CHECKING:
    from cartographer.interfaces.printer import MacroParams, Toolhead

T = TypeVar("T", bound=Enum)


def get_enum_choice(params: MacroParams, option: str, enum_type: type[T], default: T) -> T:
    choice = params.get(option, default=default.value)

    # Convert both the choice and enum values to lowercase for case-insensitive comparison
    lower_choice = str(choice).lower()
    lower_mapping = {str(v.value).lower(): v for v in enum_type}

    if lower_choice not in lower_mapping:
        msg = f"Invalid choice '{choice}' for option '{option}'"
        raise RuntimeError(msg)

    return lower_mapping[lower_choice]


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
