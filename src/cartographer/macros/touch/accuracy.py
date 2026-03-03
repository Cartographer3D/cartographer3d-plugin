from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, final

import numpy as np
from typing_extensions import override

from cartographer.interfaces.configuration import GeneralConfig
from cartographer.interfaces.printer import Macro, MacroParams
from cartographer.macros.fields import config_ref, param, parse

if TYPE_CHECKING:
    from cartographer.interfaces.printer import Toolhead
    from cartographer.probe.touch_mode import TouchMode


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TouchAccuracyParams:
    """Parameters for CARTOGRAPHER_TOUCH_ACCURACY."""

    lift_speed: float = param("Lift speed in mm/s", default=config_ref(GeneralConfig, "lift_speed"), min=1)
    sample_retract_dist: float = param("Retract distance between samples", default=1.0, min=1.0)
    samples: int = param("Number of probe samples", default=5, min=3)


@final
class TouchAccuracyMacro(Macro):
    description = "Touch the bed multiple times to measure the accuracy of the probe."

    def __init__(self, probe: TouchMode, toolhead: Toolhead, *, lift_speed: float) -> None:
        self._probe = probe
        self._toolhead = toolhead
        self._lift_speed = lift_speed

    @override
    def run(self, params: MacroParams) -> None:
        p = parse(TouchAccuracyParams, params, lift_speed=self._lift_speed)
        position = self._toolhead.get_position()

        logger.info(
            "touch accuracy at X:%.3f Y:%.3f Z:%.3f (samples=%d retract=%.3f lift_speed=%.1f)",
            position.x,
            position.y,
            position.z,
            p.samples,
            p.sample_retract_dist,
            p.lift_speed,
        )

        self._toolhead.move(z=position.z + p.sample_retract_dist, speed=p.lift_speed)
        measurements: list[float] = []
        while len(measurements) < p.samples:
            trigger_pos = self._probe.perform_probe()
            measurements.append(trigger_pos)
            pos = self._toolhead.get_position()
            self._toolhead.move(z=pos.z + p.sample_retract_dist, speed=p.lift_speed)
        logger.debug("Measurements gathered: %s", ", ".join(f"{m:.6f}" for m in measurements))

        max_value = max(measurements)
        min_value = min(measurements)
        range_value = max_value - min_value
        avg_value = np.mean(measurements)
        median = np.median(measurements)
        std_dev = np.std(measurements)

        logger.info(
            "touch accuracy results:\n"
            "maximum %.6f, minimum %.6f, range %.6f,\n"
            "average %.6f, median %.6f,\n"
            "standard deviation %.6f",
            max_value,
            min_value,
            range_value,
            avg_value,
            median,
            std_dev,
        )
