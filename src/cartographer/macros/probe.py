from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING, final

import numpy as np
from typing_extensions import override

from cartographer.interfaces.printer import Macro, MacroParams

if TYPE_CHECKING:
    from cartographer.interfaces.configuration import Configuration
    from cartographer.interfaces.printer import Toolhead
    from cartographer.probe import Probe

logger = logging.getLogger(__name__)


@final
class ProbeMacro(Macro):
    description = "Probe the bed to get the height offset at the current position."
    last_trigger_position: float | None = None

    def __init__(self, probe: Probe) -> None:
        self._probe = probe

    @override
    def run(self, params: MacroParams) -> None:
        trigger_pos = self._probe.perform_scan()
        logger.info("Result is z=%.6f", trigger_pos)
        self.last_trigger_position = trigger_pos


@final
class ProbeAccuracyMacro(Macro):
    description = "Probe the bed multiple times to measure the accuracy of the probe."

    def __init__(self, probe: Probe, toolhead: Toolhead) -> None:
        self._probe = probe
        self._toolhead = toolhead

    @override
    def run(self, params: MacroParams) -> None:
        lift_speed = params.get_float("LIFT_SPEED", 5.0, above=0)
        retract = params.get_float("SAMPLE_RETRACT_DIST", default=1.0, minval=1.0)
        sample_count = params.get_int("SAMPLES", default=10, minval=3)
        position = self._toolhead.get_position()

        logger.info(
            "probe accuracy at X:%.3f Y:%.3f Z:%.3f (samples=%d retract=%.3f lift_speed=%.1f)",
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
            trigger_pos = self._probe.perform_scan()
            measurements.append(trigger_pos)
            pos = self._toolhead.get_position()
            self._toolhead.move(z=pos.z + retract, speed=lift_speed)
        logger.debug("Measurements gathered: %s", measurements)

        max_value = max(measurements)
        min_value = min(measurements)
        range_value = max_value - min_value
        avg_value = np.mean(measurements)
        median = np.median(measurements)
        std_dev = np.std(measurements)

        logger.info(
            "probe accuracy results:\n"
            "maximum %.6f, minimum %.6f, range %.6f, average %.6f, median %.6f, standard deviation %.6f",
            max_value,
            min_value,
            range_value,
            avg_value,
            median,
            std_dev,
        )


@final
class QueryProbeMacro(Macro):
    description = "Return the status of the z-probe"
    last_triggered: bool = False

    def __init__(self, probe: Probe) -> None:
        self._probe = probe

    @override
    def run(self, params: MacroParams) -> None:
        triggered = self._probe.query_is_triggered()
        logger.info("probe: %s", "TRIGGERED" if triggered else "open")
        self.last_triggered = triggered


@final
class ZOffsetApplyProbeMacro(Macro):
    description = "Adjust the probe's z_offset"

    def __init__(self, probe: Probe, toolhead: Toolhead, config: Configuration) -> None:
        self._probe = probe
        self._toolhead = toolhead
        self._config = config

    @override
    def run(self, params: MacroParams) -> None:
        additional_offset = self._toolhead.get_gcode_z_offset()

        if self._probe.touch.last_homing_time > self._probe.scan.last_homing_time:
            self._update_touch_offset(additional_offset)
        else:
            self._update_scan_offset(additional_offset)

    def _update_touch_offset(self, additional_offset: float) -> None:
        model = self._probe.touch.get_model()
        new_offset = model.z_offset - additional_offset

        if new_offset > 0:
            logger.warning(
                "Cannot set a positive z-offset (%.3f) for %s in touch mode, offset set to 0.", new_offset, model.name
            )
            new_offset = 0

        self._config.save_touch_model(replace(model.config, z_offset=new_offset))
        self._log_offset_update("touch", model.name, new_offset)

    def _update_scan_offset(self, additional_offset: float) -> None:
        model = self._probe.scan.get_model()
        new_offset = model.z_offset - additional_offset

        self._config.save_scan_model(replace(model.config, z_offset=new_offset))
        self._log_offset_update("scan", model.name, new_offset)

    def _log_offset_update(self, mode: str, model_name: str, new_offset: float) -> None:
        logger.info(
            "cartographer: %s %s z_offset: %.3f\n"
            "The SAVE_CONFIG command will update the printer config file with the above and restart the printer.",
            mode,
            model_name,
            new_offset,
        )
