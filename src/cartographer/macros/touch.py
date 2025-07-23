from __future__ import annotations

import logging
from typing import TYPE_CHECKING, final

import numpy as np
from typing_extensions import override

from cartographer.interfaces.printer import HomingState, Macro, MacroParams
from cartographer.lib.statistics import compute_mad

if TYPE_CHECKING:
    from cartographer.interfaces.printer import Toolhead
    from cartographer.probe.touch_mode import TouchMode


logger = logging.getLogger(__name__)


@final
class TouchProbeMacro(Macro):
    description = "Touch the bed to get the height offset at the current position."
    last_trigger_position: float | None = None

    def __init__(self, probe: TouchMode) -> None:
        self._probe = probe

    @override
    def run(self, params: MacroParams) -> None:
        trigger_position = self._probe.perform_probe()
        logger.info("Result is z=%.6f", trigger_position)
        self.last_trigger_position = trigger_position


@final
class TouchAccuracyMacro(Macro):
    description = "Touch the bed multiple times to measure the accuracy of the probe."

    def __init__(self, probe: TouchMode, toolhead: Toolhead) -> None:
        self._probe = probe
        self._toolhead = toolhead

    @override
    def run(self, params: MacroParams) -> None:
        lift_speed = params.get_float("LIFT_SPEED", 5.0, above=0)
        retract = params.get_float("SAMPLE_RETRACT_DIST", 1.0, minval=0)
        sample_count = params.get_int("SAMPLES", 5, minval=1)
        position = self._toolhead.get_position()

        logger.info(
            "touch accuracy at X:%.3f Y:%.3f Z:%.3f (samples=%d retract=%.3f lift_speed=%.1f)",
            position.x,
            position.y,
            position.z,
            sample_count,
            retract,
            lift_speed,
        )

        self._toolhead.move(z=position.z + retract, speed=lift_speed)
        measurements: list[float] = []
        while len(measurements) < sample_count:
            trigger_pos = self._probe.perform_probe()
            measurements.append(trigger_pos)
            pos = self._toolhead.get_position()
            self._toolhead.move(z=pos.z + retract, speed=lift_speed)
        logger.debug("Measurements gathered: %s", ", ".join(f"{m:.6f}" for m in measurements))

        max_value = max(measurements)
        min_value = min(measurements)
        range_value = max_value - min_value
        avg_value = np.mean(measurements)
        median = np.median(measurements)
        std_dev = np.std(measurements)
        mad = compute_mad(measurements)

        logger.info(
            """
            touch accuracy results:\n
            maximum %.6f, minimum %.6f, range %.6f,\n
            average %.6f, median %.6f,\n
            standard deviation %.6f, median absolute deviation %.6f
            """,
            max_value,
            min_value,
            range_value,
            avg_value,
            median,
            std_dev,
            mad,
        )


class _FakeHomingState(HomingState):
    @override
    def is_homing_z(self) -> bool:
        return True

    @override
    def set_z_homed_position(self, position: float) -> None:
        pass


@final
class TouchHomeMacro(Macro):
    description = "Touch the bed to home Z axis"

    def __init__(
        self,
        probe: TouchMode,
        toolhead: Toolhead,
        home_position: tuple[float, float],
    ) -> None:
        self._probe = probe
        self._toolhead = toolhead
        self._home_position = home_position

    @override
    def run(self, params: MacroParams) -> None:
        if not self._toolhead.is_homed("x") or not self._toolhead.is_homed("y"):
            msg = "Must home x and y before touch homing"
            raise RuntimeError(msg)

        forced_z = False
        if not self._toolhead.is_homed("z"):
            forced_z = True
            _, z_max = self._toolhead.get_axis_limits("z")
            self._toolhead.set_z_position(z=z_max - 10)

        pos = self._toolhead.get_position()
        # TODO: Get rid of magic constants
        self._toolhead.move(
            z=pos.z + 2,
            speed=5,
        )
        self._toolhead.move(
            x=self._home_position[0],
            y=self._home_position[1],
            speed=50,
        )
        self._toolhead.wait_moves()

        try:
            trigger_pos = self._probe.perform_probe()
        finally:
            if forced_z:
                self._toolhead.clear_z_homing_state()

        pos = self._toolhead.get_position()
        self._toolhead.set_z_position(pos.z - trigger_pos)
        # Simulate home end to update last homing time
        self._probe.on_home_end(_FakeHomingState())
        logger.info(
            "Touch home at (%.3f,%.3f) adjusted z by %.3f",
            pos.x,
            pos.y,
            trigger_pos,
        )
