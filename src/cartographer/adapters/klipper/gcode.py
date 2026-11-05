from __future__ import annotations

from typing import TYPE_CHECKING, cast, final

from typing_extensions import override

from cartographer.interfaces.printer import GCodeDispatch, MacroParams

if TYPE_CHECKING:
    from gcode import GCodeCommand
    from klippy import Printer


@final
class KlipperGCodeDispatch(GCodeDispatch):
    def __init__(self, printer: Printer) -> None:
        self._gcode = printer.lookup_object("gcode")

    @override
    def run_gcode(self, script: str, *, gcmd: MacroParams) -> None:
        raw_gcmd = cast("GCodeCommand", gcmd)
        if callable(getattr(raw_gcmd, "run_script", None)):
            return raw_gcmd.run_script(script)
        return self._gcode.run_script_from_command(script)
