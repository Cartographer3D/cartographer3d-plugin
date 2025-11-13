# https://github.com/Klipper3d/klipper/blob/master/klippy/gcode.py
from collections.abc import Callable
from typing import NamedTuple, overload

class CommandError(Exception): ...

class GCodeCommand:
    error: type[CommandError]

    def respond_raw(self, msg: str) -> None: ...
    def respond_info(self, msg: str, log: bool = True) -> None: ...
    def get_command_parameters(self) -> dict[str, str]: ...
    @overload
    def get(
        self,
        name: str,
        default: str = ...,
    ) -> str: ...
    @overload
    def get(
        self,
        name: str,
        default: None,
    ) -> str | None: ...
    @overload
    def get_int(
        self,
        name: str,
        default: int = ...,
        minval: int | None = None,
        maxval: int | None = None,
    ) -> int: ...
    @overload
    def get_int(
        self,
        name: str,
        default: None,
        minval: int | None = None,
        maxval: int | None = None,
    ) -> int | None: ...
    @overload
    def get_float(
        self,
        name: str,
        default: float = ...,
        minval: float | None = None,
        maxval: float | None = None,
        above: float | None = None,
        below: float | None = None,
    ) -> float: ...
    @overload
    def get_float(
        self,
        name: str,
        default: None,
        minval: float | None = None,
        maxval: float | None = None,
        above: float | None = None,
        below: float | None = None,
    ) -> float | None: ...

class Coord(NamedTuple):
    x: float
    y: float
    z: float
    e: float

class GCodeDispatch:
    error: type[CommandError]
    Coord: type[Coord]

    def respond_raw(self, msg: str) -> None: ...
    def respond_info(self, msg: str, log: bool = True) -> None: ...
    def run_script_from_command(self, script: str) -> None: ...
    @overload
    def register_command(
        self,
        cmd: str,
        func: Callable[[GCodeCommand], None],
        when_not_ready: bool = False,
        desc: str | None = None,
    ) -> None: ...
    @overload
    def register_command(
        self,
        cmd: str,
        func: None,
        when_not_ready: bool = False,
        desc: str | None = None,
    ) -> Callable[[GCodeCommand], None] | None: ...
    def create_gcode_command(
        self,
        command: str,
        commandline: str,
        params: dict[str, str],
    ) -> GCodeCommand: ...
