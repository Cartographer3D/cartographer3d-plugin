from __future__ import annotations

from typing import TYPE_CHECKING, final

from gcode import GCodeCommand
from typing_extensions import override

from cartographer.interfaces.printer import GCodeDispatch, MacroParams

if TYPE_CHECKING:
    from klippy import Printer


@final
class KlipperGCodeDispatch(GCodeDispatch):
    def __init__(self, printer: Printer) -> None:
        self._gcode = printer.lookup_object("gcode")

    @override
    def run_gcode(self, script: str) -> None:
        return self._gcode.run_script_from_command(script)

    @override
    def clone_params(self, params: MacroParams, overrides: dict[str, str]) -> MacroParams:
        assert isinstance(params, GCodeCommand), f"Invalid params type {type(params).__name__}, expected GCodeCommand"

        # Merge original parameters with overrides
        new_params = params.get_command_parameters().copy()
        new_params.update(overrides)

        # Rebuild command line with all parameters
        param_string = " ".join(f"{k}={v}" for k, v in new_params.items())
        new_commandline = f"{params.get_command()} {param_string}"

        # Create new GCodeCommand with updated parameters
        return self._gcode.create_gcode_command(
            command=params.get_command(),
            commandline=new_commandline,
            params=new_params,
        )
