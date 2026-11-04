from __future__ import annotations

import inspect
import logging
from abc import ABC
from typing import TYPE_CHECKING, Callable, final

from extras.homing import Homing
from extras.manual_probe import ManualProbeHelper
from typing_extensions import override

from cartographer.adapters.klipper.endstop import KlipperEndstop
from cartographer.adapters.klipper_like.axis_compat import uses_string_homing_axes
from cartographer.adapters.klipper_like.gcode_context import gcode_command_context, get_gcode_command
from cartographer.adapters.klipper_like.utils import reraise_from_klipper
from cartographer.interfaces.printer import Endstop, HomingAxis, Position, TemperatureStatus, Toolhead

if TYPE_CHECKING:
    from configfile import ConfigWrapper
    from gcode import GCodeCommand
    from klippy import Printer
    from mcu import MCU_endstop
    from stepper import MCU_stepper
    from toolhead import ToolHead as KlippyToolhead

    from cartographer.mcu.mcu import CartographerMcu

logger = logging.getLogger(__name__)

axis_mapping: dict[HomingAxis, int] = {
    "x": 0,
    "y": 1,
    "z": 2,
}


def axis_to_index(axis: HomingAxis) -> int:
    return axis_mapping[axis]


@final
class FakeRail:
    def __init__(self, endstop: MCU_endstop, axis_range: tuple[float, float]) -> None:
        self.endstop = endstop
        self.axis_range = axis_range

    def get_range(self) -> tuple[float, float]:
        return self.axis_range

    def get_name(self, short: bool = False) -> str:
        if short:
            return "carto"
        return "cartographer"

    def get_steppers(self) -> list[MCU_stepper]:
        return self.endstop.get_steppers()

    def get_endstops(self) -> list[tuple[MCU_endstop, str]]:
        return [(self.endstop, "cartographer")]


class KlipperLikeToolhead(Toolhead, ABC):
    __toolhead: KlippyToolhead | None = None
    __use_str_axes: bool | None = None

    @property
    def toolhead(self) -> KlippyToolhead:
        if self.__toolhead is None:
            self.__toolhead = self.printer.lookup_object("toolhead")
        return self.__toolhead

    @property
    def _use_str_axes(self) -> bool:
        if self.__use_str_axes is None:
            self.__use_str_axes = uses_string_homing_axes(self.toolhead)
        return self.__use_str_axes

    def __init__(self, config: ConfigWrapper, mcu: CartographerMcu) -> None:
        self.mcu: CartographerMcu = mcu
        self.printer: Printer = config.get_printer()

    @override
    def get_last_move_time(self) -> float:
        return self.toolhead.get_last_move_time()

    @override
    def wait_moves(self) -> None:
        self.toolhead.wait_moves()

    @override
    def get_position(self) -> Position:
        pos = self.toolhead.get_position()
        return Position(x=pos[0], y=pos[1], z=pos[2])

    @override
    def move(self, *, x: float | None = None, y: float | None = None, z: float | None = None, speed: float) -> None:
        self.toolhead.manual_move([x, y, z], speed=speed)

    @override
    def is_homed(self, axis: HomingAxis) -> bool:
        time = self.mcu.get_current_time()
        return axis in self.toolhead.get_status(time)["homed_axes"]

    @override
    def get_gcode_z_offset(self) -> float:
        gcode_move = self.printer.lookup_object("gcode_move")
        return gcode_move.get_status()["homing_origin"].z

    @override
    @reraise_from_klipper
    def z_probing_move(self, endstop: Endstop, *, speed: float) -> float:
        probing_move = self.printer.lookup_object("homing").probing_move
        requires_gcmd = "gcmd" in inspect.signature(probing_move).parameters
        gcmd = get_gcode_command()
        if requires_gcmd and gcmd is None:
            msg = "Probing requires an active G-code command context"
            raise RuntimeError(msg)

        klipper_endstop = KlipperEndstop(self.mcu, endstop)
        self.wait_moves()
        z_min, _ = self.get_axis_limits("z")

        pos = self.toolhead.get_position()[:]
        pos[2] = z_min

        if requires_gcmd:
            epos = probing_move(klipper_endstop, pos, speed, gcmd=gcmd)
        else:
            epos = probing_move(klipper_endstop, pos, speed)
        return epos[2]

    @override
    def z_home_end(self, endstop: Endstop) -> None:
        requires_gcmd = "gcmd" in inspect.signature(Homing).parameters
        gcmd = get_gcode_command()
        if requires_gcmd and gcmd is None:
            msg = "Homing completion requires an active G-code command context"
            raise RuntimeError(msg)

        klipper_endstop = KlipperEndstop(self.mcu, endstop)
        homing = Homing(self.printer, gcmd=gcmd) if requires_gcmd else Homing(self.printer)
        homing.set_axes([axis_to_index("z")])
        homing.trigger_mcu_pos = {sp.get_name(): sp.get_mcu_position() for sp in klipper_endstop.get_steppers()}
        homing.homing_successful = True

        self.printer.send_event("homing:home_rails_end", homing, [FakeRail(klipper_endstop, self.get_axis_limits("z"))])

    @override
    def set_z_position(self, z: float) -> None:
        pos = self.toolhead.get_position()[:]
        pos[2] = z

        homing_axes = "z" if self._use_str_axes else (2,)
        self.toolhead.set_position(pos, homing_axes)

    @override
    def get_axis_limits(self, axis: HomingAxis) -> tuple[float, float]:
        time = self.toolhead.get_last_move_time()
        status = self.toolhead.get_status(time)
        index = axis_to_index(axis)
        return status["axis_minimum"][index], status["axis_maximum"][index]

    @override
    def manual_probe(self, finalize_callback: Callable[[Position | None], None]) -> None:
        passes_gcmd = "pass_gcmd" in inspect.signature(ManualProbeHelper).parameters
        if passes_gcmd:
            gcmd = get_gcode_command()
            if gcmd is None:
                msg = "Manual probing requires an active G-code command context"
                raise RuntimeError(msg)
        else:
            gcode = self.printer.lookup_object("gcode")
            gcmd = gcode.create_gcode_command("", "", {})

        def finalize(pos: list[float] | None, callback_gcmd: GCodeCommand | None = None) -> None:
            with gcode_command_context(callback_gcmd):
                finalize_callback(Position(pos[0], pos[1], pos[2]) if pos else None)

        if passes_gcmd:
            _ = ManualProbeHelper(self.printer, gcmd, finalize, pass_gcmd=True)
        else:
            _ = ManualProbeHelper(self.printer, gcmd, finalize)

    @override
    def clear_z_homing_state(self) -> None:
        if not self._use_str_axes:
            self.toolhead.get_kinematics().note_z_not_homed()
            return

        self.toolhead.get_kinematics().clear_homing_state("z")

    @override
    def dwell(self, seconds: float) -> None:
        self.toolhead.dwell(seconds)

    @override
    def get_extruder_temperature(self) -> TemperatureStatus:
        time = self.mcu.get_current_time()
        heater = self.toolhead.get_extruder().get_heater().get_status(time)
        return TemperatureStatus(heater["temperature"], heater["target"])
