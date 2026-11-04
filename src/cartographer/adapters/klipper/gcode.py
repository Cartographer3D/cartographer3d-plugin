from __future__ import annotations

from typing import TYPE_CHECKING, final

from typing_extensions import override

from cartographer.adapters.klipper_like.gcode_context import get_gcode_command
from cartographer.interfaces.printer import GCodeDispatch

if TYPE_CHECKING:
    from klippy import Printer


@final
class KlipperGCodeDispatch(GCodeDispatch):
    def __init__(self, printer: Printer) -> None:
        self._gcode = printer.lookup_object("gcode")

    @override
    def run_gcode(self, script: str) -> None:
        gcmd = get_gcode_command()
        if gcmd is not None and callable(getattr(gcmd, "run_script", None)):
            return gcmd.run_script(script)
        return self._gcode.run_script_from_command(script)
