from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Callable, Sequence, TypeVar

from gcode import CommandError
from gcode import Coord as GCodeCoord
from typing_extensions import ParamSpec

from cartographer.interfaces.errors import PrinterShutdownError, ProbeTriggerError

if TYPE_CHECKING:
    from cartographer.interfaces.printer import Position

if TYPE_CHECKING:
    from configfile import ConfigWrapper
    from klippy import Printer

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
        except PrinterShutdownError:
            msg = "Aborted: printer entered shutdown"
            raise CommandError(msg) from None
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


def try_load_object(printer: Printer, config: ConfigWrapper, section: str) -> bool:
    if not config.has_section(section):
        return False
    _ = printer.load_object(config.getsection(section), section)
    return True


def build_probe_method_macros(
    printer: Printer,
    config: ConfigWrapper,
    z_tilt_sections: Sequence[str] = ("z_tilt",),
) -> list[str]:
    """Build the probe-method macro list from printer configuration.

    Always includes BED_MESH_CALIBRATE. Adds Z_TILT_ADJUST if any of
    *z_tilt_sections* is configured (at most once). Adds QUAD_GANTRY_LEVEL
    and SCREWS_TILT_CALCULATE when the corresponding sections are present.
    """
    macros: list[str] = ["BED_MESH_CALIBRATE"]
    if any(try_load_object(printer, config, s) for s in z_tilt_sections):
        macros.append("Z_TILT_ADJUST")
    if try_load_object(printer, config, "quad_gantry_level"):
        macros.append("QUAD_GANTRY_LEVEL")
    if try_load_object(printer, config, "screws_tilt_adjust"):
        macros.append("SCREWS_TILT_CALCULATE")
    return macros
