from __future__ import annotations

import logging
from abc import ABC
from typing import TYPE_CHECKING, Callable, final

from extras.homing import Homing
from extras.manual_probe import ManualProbeHelper
from typing_extensions import override

from cartographer.adapters.klipper.endstop import KlipperEndstop
from cartographer.adapters.klipper_like.utils import reraise_from_klipper
from cartographer.interfaces.printer import Endstop, HomingAxis, Position, TemperatureStatus, Toolhead

if TYPE_CHECKING:
    from configfile import ConfigWrapper
    from klippy import Printer
    from mcu import MCU_endstop
    from stepper import MCU_stepper
    from toolhead import ToolHead as KlippyToolhead

    from cartographer.adapters.klipper.mcu.mcu import KlipperCartographerMcu

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
    def __init__(self, endstop: MCU_endstop) -> None:
        self.endstop = endstop

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
            kin = self.toolhead.get_kinematics()
            self.__use_str_axes = not hasattr(kin, "note_z_not_homed")
        return self.__use_str_axes

    def __init__(self, config: ConfigWrapper, mcu: KlipperCartographerMcu) -> None:
        self.mcu: KlipperCartographerMcu = mcu
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
        klipper_endstop = KlipperEndstop(self.mcu, endstop)
        self.wait_moves()
        z_min, _ = self.get_axis_limits("z")

        pos = self.toolhead.get_position()[:]
        pos[2] = z_min

        epos = self.printer.lookup_object("homing").probing_move(klipper_endstop, pos, speed)
        return epos[2]

    @override
    def z_home_end(self, endstop: Endstop) -> None:
        klipper_endstop = KlipperEndstop(self.mcu, endstop)

        homing = Homing(self.printer)
        homing.set_axes([axis_to_index("z")])
        homing.trigger_mcu_pos = {sp.get_name(): sp.get_mcu_position() for sp in klipper_endstop.get_steppers()}

        self.printer.send_event("homing:home_rails_end", homing, [FakeRail(klipper_endstop)])

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
        gcode = self.printer.lookup_object("gcode")
        gcmd = gcode.create_gcode_command("", "", {})
        _ = ManualProbeHelper(
            self.printer,
            gcmd,
            lambda pos: finalize_callback(Position(pos[0], pos[1], pos[2]) if pos else None),
        )

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
