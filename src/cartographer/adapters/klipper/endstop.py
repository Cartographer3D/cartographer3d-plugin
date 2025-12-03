from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast, final

from mcu import MCU_endstop
from typing_extensions import override

from cartographer.adapters.klipper_like.utils import reraise_for_klipper
from cartographer.interfaces.printer import HomingAxis, HomingState

if TYPE_CHECKING:
    from extras.homing import Homing
    from mcu import MCU
    from reactor import ReactorCompletion
    from stepper import MCU_stepper

    from cartographer.adapters.klipper.mcu.mcu import KlipperCartographerMcu
    from cartographer.interfaces.printer import Endstop

logger = logging.getLogger(__name__)

axis_mapping: dict[HomingAxis, int] = {
    "x": 0,
    "y": 1,
    "z": 2,
}


def axis_to_index(axis: HomingAxis) -> int:
    return axis_mapping[axis]


@final
class KlipperHomingState(HomingState):
    def __init__(self, homing: Homing) -> None:
        self.homing = homing

    @override
    def is_homing_z(self) -> bool:
        return axis_to_index("z") in self.homing.get_axes()

    @override
    def set_z_homed_position(self, position: float) -> None:
        logger.debug("Setting homed distance for z to %.3f", position)
        self.homing.set_homed_position([None, None, position])


@final
class KlipperEndstop(MCU_endstop):
    def __init__(self, mcu: KlipperCartographerMcu, endstop: Endstop):
        self.mcu = mcu
        self.endstop = endstop

    @override
    def get_mcu(self) -> MCU:
        return self.mcu.klipper_mcu

    @override
    def add_stepper(self, stepper: MCU_stepper) -> None:
        return self.mcu.dispatch.add_stepper(stepper)

    @override
    def get_steppers(self) -> list[MCU_stepper]:
        return self.mcu.dispatch.get_steppers()

    @override
    @reraise_for_klipper
    def home_start(
        self,
        print_time: float,
        sample_time: float,
        sample_count: int,
        rest_time: float,
        triggered: bool = True,
    ) -> ReactorCompletion:
        del sample_time, sample_count, rest_time, triggered
        return cast("ReactorCompletion", self.endstop.home_start(print_time))

    @override
    @reraise_for_klipper
    def home_wait(self, home_end_time: float) -> float:
        return self.endstop.home_wait(home_end_time)

    @override
    @reraise_for_klipper
    def query_endstop(self, print_time: float) -> int:
        return 1 if self.endstop.query_is_triggered(print_time) else 0

    @override
    def get_position_endstop(self) -> float:
        return self.endstop.get_endstop_position()
