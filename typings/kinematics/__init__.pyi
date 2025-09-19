# https://github.com/Klipper3d/klipper/blob/master/klippy/kinematics/none.py
from typing import Protocol, TypedDict

import gcode
from stepper import MCU_stepper

# NOTE: This is a partial definition of the Kinematics class
# This module does not exist in the real Klipper
# DO NOT IMPORT THIS MODULE

type _Pos = list[float]

class Status(TypedDict):
    homed_axes: str
    axis_minimum: gcode.Coord
    axis_maximum: gcode.Coord

class Kinematics(Protocol):
    def get_steppers(self) -> list[MCU_stepper]: ...
    def get_status(self, eventtime: float) -> Status: ...
    def calc_position(self, stepper_positions: dict[str, float]) -> _Pos: ...
    def clear_homing_state(
        self, axes: str | tuple[int, ...]
    ) -> None: ...  # TODO: Kalico and old klippy takes tuple[int,..]
    def note_z_not_homed(self) -> None: ...  # TODO: Kalico and old klippy has this method
