from __future__ import annotations

import logging
import math
from random import random
from typing import TYPE_CHECKING, final

from typing_extensions import override

from cartographer.interfaces.printer import Macro, MacroParams
from cartographer.macros.utils import forced_z

if TYPE_CHECKING:
    from cartographer.interfaces.printer import Toolhead
    from cartographer.probe.touch_mode import TouchMode


logger = logging.getLogger(__name__)


@final
class TouchHomeMacro(Macro):
    description = "Touch the bed to home Z axis"

    def __init__(
        self,
        probe: TouchMode,
        toolhead: Toolhead,
        *,
        home_position: tuple[float, float],
        travel_speed: float,
        random_radius: float,
    ) -> None:
        self._probe = probe
        self._toolhead = toolhead
        self._home_position = home_position
        self._travel_speed = travel_speed
        self._random_radius = random_radius

    @override
    def run(self, params: MacroParams) -> None:
        random_radius = params.get_float("EXPERIMENTAL_RANDOM_RADIUS", default=self._random_radius, minval=0)
        if not self._toolhead.is_homed("x") or not self._toolhead.is_homed("y"):
            msg = "Must home x and y before touch homing"
            raise RuntimeError(msg)

        # Capture the starting Z position before forced_z modifies it
        initial_z = self._toolhead.get_position().z if self._toolhead.is_homed("z") else None

        with forced_z(self._toolhead):
            pos = self._toolhead.get_position()
            # TODO: Get rid of magic constants
            self._toolhead.move(
                z=pos.z + 2,
                speed=5,
            )
            home_x, home_y = self._get_homing_position(random_radius)
            self._toolhead.move(
                x=home_x,
                y=home_y,
                speed=self._travel_speed,
            )
            self._toolhead.wait_moves()

            trigger_pos = self._probe.perform_probe()

        self._toolhead.z_home_end(self._probe)
        pos = self._toolhead.get_position()
        self._toolhead.set_z_position(pos.z - trigger_pos)

        # Calculate the actual adjustment relative to where we started
        # If Z was already homed, show adjustment from that position
        # If Z was not homed, show the raw trigger position
        if initial_z is not None:
            actual_adjustment = initial_z - (pos.z - trigger_pos)
            logger.info(
                "Touch home at (%.3f,%.3f) adjusted z by %.3f",
                pos.x,
                pos.y,
                actual_adjustment,
            )
        else:
            logger.info(
                "Touch home at (%.3f,%.3f) set z to %.3f",
                pos.x,
                pos.y,
                pos.z - trigger_pos,
            )

    def _get_homing_position(self, random_radius: float) -> tuple[float, float]:
        center_x, center_y = self._home_position
        u1 = random()  # [0, 1)
        u2 = random()  # [0, 1)

        # Polar coordinates with square root for uniform area distribution
        radius = random_radius * math.sqrt(u1)
        angle = 2 * math.pi * u2

        # Convert to Cartesian coordinates
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)

        return (x, y)
