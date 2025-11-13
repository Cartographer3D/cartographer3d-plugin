# Helper script to determine a Z height
from collections.abc import Callable

from gcode import GCodeCommand
from klippy import Printer

type _Pos = list[float]

def verify_no_manual_probe(printer: Printer) -> None: ...

class ManualProbeHelper:
    def __init__(
        self,
        printer: Printer,
        gcmd: GCodeCommand,
        finalize_callback: Callable[[_Pos | None], None],
    ) -> None: ...
