# https://github.com/Klipper3d/klipper/blob/master/klippy/klippy.py
from collections.abc import Callable
from typing import Literal, overload

import configfile
from cartographer.core import PrinterCartographer
from configfile import ConfigWrapper, PrinterConfig
from extras.axis_twist_compensation import AxisTwistCompensation
from extras.bed_mesh import BedMesh
from extras.exclude_object import ExcludeObject
from extras.heaters import PrinterHeaters
from extras.homing import Homing, HomingMove, PrinterHoming
from extras.manual_probe import ProbeResult
from extras.motion_report import PrinterMotionReport
from gcode import CommandError, GCodeDispatch
from gcode_move import GCodeMove
from pins import PrinterPins
from reactor import Reactor
from stepper import GenericPrinterRail
from toolhead import ToolHead

# TODO: Kalico specific
APP_NAME: str

class Printer:
    config_error: type[configfile.error]
    command_error: type[CommandError]
    def add_object(self, name: str, obj: object) -> None: ...
    @overload
    def load_object(
        self,
        config: ConfigWrapper,
        section: Literal["bed_mesh"],
    ) -> BedMesh: ...
    @overload
    def load_object(
        self,
        config: ConfigWrapper,
        section: Literal["heaters"],
    ) -> PrinterHeaters: ...
    @overload
    def load_object(
        self,
        config: ConfigWrapper,
        section: Literal["motion_report"],
    ) -> PrinterMotionReport: ...
    @overload
    def load_object(
        self,
        config: ConfigWrapper,
        section: Literal["axis_twist_compensation"],
    ) -> AxisTwistCompensation: ...
    def is_shutdown(self) -> bool: ...
    def invoke_shutdown(self, msg: str) -> None: ...
    def get_reactor(self) -> Reactor: ...
    @overload
    def register_event_handler(self, event: Literal["klippy:connect"], callback: Callable[[], None]) -> None: ...
    @overload
    def register_event_handler(self, event: Literal["klippy:ready"], callback: Callable[[], None]) -> None: ...
    @overload
    def register_event_handler(self, event: Literal["klippy:disconnect"], callback: Callable[[], None]) -> None: ...
    @overload
    def register_event_handler(self, event: Literal["klippy:shutdown"], callback: Callable[[], None]) -> None: ...
    @overload
    def register_event_handler(self, event: Literal["klippy:mcu_identify"], callback: Callable[[], None]) -> None: ...
    @overload
    def register_event_handler(
        self,
        event: Literal["homing:home_rails_begin"],
        callback: Callable[[Homing, list[GenericPrinterRail]], None],
    ) -> None: ...
    @overload
    def register_event_handler(
        self,
        event: Literal["homing:home_rails_end"],
        callback: Callable[[Homing, list[GenericPrinterRail]], None],
    ) -> None: ...
    @overload
    def register_event_handler(
        self,
        event: Literal["homing:homing_move_begin"],
        callback: Callable[[HomingMove], None],
    ) -> None: ...
    @overload
    def register_event_handler(
        self,
        event: Literal["homing:homing_move_end"],
        callback: Callable[[HomingMove], None],
    ) -> None: ...
    @overload
    def lookup_object(self, name: Literal["exclude_object"], default: None) -> ExcludeObject | None: ...
    @overload
    def lookup_object(self, name: Literal["bed_mesh"]) -> BedMesh: ...
    @overload
    def lookup_object(self, name: Literal["configfile"]) -> PrinterConfig: ...
    @overload
    def lookup_object(self, name: Literal["gcode"]) -> GCodeDispatch: ...
    @overload
    def lookup_object(self, name: Literal["gcode_move"]) -> GCodeMove: ...
    @overload
    def lookup_object(self, name: Literal["homing"]) -> PrinterHoming: ...
    @overload
    def lookup_object(self, name: Literal["motion_report"]) -> PrinterMotionReport: ...
    @overload
    def lookup_object(self, name: Literal["pins"]) -> PrinterPins: ...
    @overload
    def lookup_object(self, name: Literal["toolhead"]) -> ToolHead: ...
    @overload
    def lookup_object(self, name: Literal["cartographer"]) -> PrinterCartographer: ...
    @overload
    def send_event(self, event: Literal["probe:update_results"], pos: list[float] | list[ProbeResult]) -> None: ...
    @overload
    def send_event(
        self, event: Literal["homing:home_rails_end"], homing: Homing, rails: list[GenericPrinterRail]
    ) -> None: ...
