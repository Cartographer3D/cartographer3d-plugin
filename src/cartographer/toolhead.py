from __future__ import annotations

from typing import Callable

from typing_extensions import override

from cartographer.interfaces.printer import Endstop, HomingAxis, Position, TemperatureStatus, Toolhead


class BacklashCompensatingToolhead(Toolhead):
    def __init__(self, toolhead: Toolhead, backlash: float):
        self.toolhead: Toolhead = toolhead
        self.backlash: float = backlash

    @override
    def move(self, *, x: float | None = None, y: float | None = None, z: float | None = None, speed: float) -> None:
        current_z = self.toolhead.get_position().z
        if z is None or z <= current_z:
            # Moving down or horizontally — no compensation needed
            self.toolhead.move(x=x, y=y, z=z, speed=speed)
            return

        self.toolhead.move(x=x, y=y, z=z + self.backlash, speed=speed)
        self.toolhead.move(x=x, y=y, z=z, speed=speed)

    @override
    def get_position(self) -> Position:
        return self.toolhead.get_position()

    @override
    def get_last_move_time(self) -> float:
        return self.toolhead.get_last_move_time()

    @override
    def wait_moves(self) -> None:
        self.toolhead.wait_moves()

    @override
    def is_homed(self, axis: HomingAxis) -> bool:
        return self.toolhead.is_homed(axis)

    @override
    def get_gcode_z_offset(self) -> float:
        return self.toolhead.get_gcode_z_offset()

    @override
    def z_probing_move(self, endstop: Endstop, *, speed: float) -> float:
        return self.toolhead.z_probing_move(endstop, speed=speed)

    @override
    def z_home_end(self, endstop: Endstop) -> None:
        return self.toolhead.z_home_end(endstop)

    @override
    def set_z_position(self, z: float) -> None:
        self.toolhead.set_z_position(z)

    @override
    def get_axis_limits(self, axis: HomingAxis) -> tuple[float, float]:
        return self.toolhead.get_axis_limits(axis)

    @override
    def manual_probe(self, finalize_callback: Callable[[Position | None], None]) -> None:
        self.toolhead.manual_probe(finalize_callback)

    @override
    def clear_z_homing_state(self) -> None:
        self.toolhead.clear_z_homing_state()

    @override
    def dwell(self, seconds: float) -> None:
        self.toolhead.dwell(seconds)

    @override
    def get_extruder_temperature(self) -> TemperatureStatus:
        return self.toolhead.get_extruder_temperature()

    @override
    def get_max_accel(self) -> float:
        return self.toolhead.get_max_accel()

    @override
    def set_max_accel(self, accel: float) -> None:
        self.toolhead.set_max_accel(accel)
