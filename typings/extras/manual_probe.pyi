# Helper script to determine a Z height
from collections.abc import Callable
from typing import Literal, NamedTuple, overload

from gcode import GCodeCommand
from klippy import Printer

type _Pos = list[float]

def verify_no_manual_probe(printer: Printer) -> None: ...

class ManualProbeHelper:
    @overload
    def __init__(
        self,
        printer: Printer,
        gcmd: GCodeCommand,
        finalize_callback: Callable[[_Pos | None], None],
        pass_gcmd: Literal[False] = False,
    ) -> None: ...
    @overload
    def __init__(
        self,
        printer: Printer,
        gcmd: GCodeCommand,
        finalize_callback: Callable[[_Pos | None, GCodeCommand], None],
        pass_gcmd: Literal[True],
    ) -> None: ...

class ProbeResult(NamedTuple):
    """
    WARNING: This class may not exist at runtime depending on klipper version.
    Always check with hasattr() before use.

    Example:
        if hasattr(manual_probe, 'ProbeResult'):
            result = manual_probe.ProbeResult(...)
    """

    bed_x: float
    bed_y: float
    bed_z: float
    test_x: float
    test_y: float
    test_z: float
