# https://github.com/Klipper3d/klipper/blob/master/klippy/extras/homing.py

from typing import Protocol

from mcu import MCU_endstop
from stepper import MCU_stepper

type _Pos = list[float]

class _McuProbe(Protocol):
    def get_steppers(self) -> list[MCU_stepper]: ...

class PrinterHoming:
    def probing_move(self, mcu_probe: _McuProbe, pos: _Pos, speed: float) -> _Pos: ...

class Homing:
    def set_homed_position(self, pos: list[float | None]) -> None: ...
    def get_axes(self) -> list[int]: ...

class HomingMove:
    def __init__(self, endstops: list[tuple[MCU_endstop, str]]) -> None: ...
    def homing_move(self, pos: _Pos, speed: float, probe_pos: bool) -> _Pos: ...
