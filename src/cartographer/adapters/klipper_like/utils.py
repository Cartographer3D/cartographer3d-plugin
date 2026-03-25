from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Callable, TypeVar

from gcode import CommandError
from gcode import Coord as GCodeCoord
from typing_extensions import ParamSpec

from cartographer.interfaces.errors import ProbeTriggerError

if TYPE_CHECKING:
    from cartographer.interfaces.printer import Position

P = ParamSpec("P")
R = TypeVar("R")

# Klipper error message for probe triggered before movement
PROBE_TRIGGERED_BEFORE_MOVEMENT = "Probe triggered prior to movement"


def make_coord(position: Position | None) -> GCodeCoord:
    """Create a GCodeCoord from a Position, or a zero Coord if None.

    Handles both old Klipper/Kalico (namedtuple with positional args)
    and new Klipper (tuple subclass accepting a single iterable).
    """
    if position:
        x, y, z = position.as_tuple()
        t = (round(x, 6), round(y, 6), round(z, 6))
    else:
        t = (0, 0, 0)
    try:
        return GCodeCoord(t)
    except TypeError:
        return GCodeCoord(t[0], t[1], t[2], 0)


def reraise_for_klipper(
    func: Callable[P, R],
) -> Callable[P, R]:
    """
    Convert RuntimeError to CommandError for Klipper compatibility.

    Use this decorator on methods that are called by Klipper and may
    raise RuntimeError.  Klipper expects CommandError for user-facing
    errors.
    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            raise CommandError(str(e)) from e

    return wrapper


def reraise_from_klipper(
    func: Callable[P, R],
) -> Callable[P, R]:
    """
    Convert Klipper CommandError to RuntimeError for internal use.

    Use this decorator on methods that call into Klipper code which
    may raise CommandError. Our internal code expects RuntimeError.
    """
    from gcode import CommandError

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return func(*args, **kwargs)
        except CommandError as e:
            error_message = str(e)
            if error_message == PROBE_TRIGGERED_BEFORE_MOVEMENT:
                raise ProbeTriggerError(error_message) from e
            raise RuntimeError(error_message) from e

    return wrapper
