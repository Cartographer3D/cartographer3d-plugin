from __future__ import annotations

import logging
from typing import TYPE_CHECKING, final

import numpy as np
from typing_extensions import override

from cartographer.interfaces.printer import Macro, MacroParams, Mcu

if TYPE_CHECKING:
    from cartographer.interfaces.printer import Toolhead
    from cartographer.probe.scan_mode import ScanMode

logger = logging.getLogger(__name__)


@final
class ScanAccuracyMacro(Macro):
    description = "Collect samples from the probe and calculate statistics on the results."

    def __init__(self, scan: ScanMode, toolhead: Toolhead, mcu: Mcu) -> None:
        self._scan = scan
        self._toolhead = toolhead
        self._mcu = mcu

    @override
    def run(self, params: MacroParams) -> None:
        readings = params.get_int("READINGS", 20, minval=10)
        sample_count = params.get_int("SAMPLES", 100, minval=10)
        position = self._toolhead.get_position()

        logger.info(
            "scan accuracy at X:%.3f Y:%.3f Z:%.3f (samples=%d)",
            position.x,
            position.y,
            position.z,
            sample_count,
        )

        measurements: list[float] = []
        while len(measurements) < sample_count:
            dist = self._scan.measure_distance(min_sample_count=readings)
            measurements.append(dist)
        logger.debug("Measurements gathered: %s", measurements)

        max_value = max(measurements)
        min_value = min(measurements)
        range_value = max_value - min_value
        avg_value = np.mean(measurements)
        median = np.median(measurements)
        std_dev = np.std(measurements)

        logger.info(
            "scan accuracy results:\n"
            "maximum %.6f, minimum %.6f, range %.6f, average %.6f, median %.6f, standard deviation %.6f",
            max_value,
            min_value,
            range_value,
            avg_value,
            median,
            std_dev,
        )
