from __future__ import annotations

import logging
from typing import TYPE_CHECKING, final

if TYPE_CHECKING:
    from cartographer.adapters.klipper.mcu.mcu import KlipperCartographerMcu
    from cartographer.interfaces.configuration import CoilConfiguration
    from cartographer.interfaces.printer import Sample

logger = logging.getLogger(__name__)

REPORT_TIME = 0.300
ABSOLUTE_ZERO_TEMP = -273.15  # Celsius
ARBITRARY_MAX_TEMP = 9999.0


@final
class PrinterTemperatureCoil:
    last_temp = 0.0
    measured_min = ARBITRARY_MAX_TEMP
    measured_max = ABSOLUTE_ZERO_TEMP
    temperature_warning = False

    def __init__(self, mcu: KlipperCartographerMcu, config: CoilConfiguration) -> None:
        self.mcu = mcu
        self.name = config.name

        self.min_temp = config.min_temp
        self.max_temp = config.max_temp
        self.mcu.register_callback(self._sample_callback)

    def get_report_time_delta(self) -> float:
        return REPORT_TIME

    def _sample_callback(self, sample: Sample) -> None:
        temp = sample.temperature
        is_out_of_range = not (self.min_temp <= temp <= self.max_temp)

        if is_out_of_range and not self.temperature_warning:
            self.temperature_warning = True
            logger.warning(
                "temperature for %(sensor_name)s at %(temperature)s is out of range [%(min_temp)s, %(max_temp)s]",
                dict(
                    sensor_name=self.name,
                    temperature=sample.temperature,
                    min_temp=self.min_temp,
                    max_temp=self.max_temp,
                ),
            )

        self.last_temp = temp
        self.measured_min = min(self.measured_min, temp)
        self.measured_max = max(self.measured_max, temp)

    def get_temp(self, eventtime: float):
        del eventtime
        return self.last_temp, 0.0

    def stats(self, eventtime: float):
        del eventtime
        return False, f"{self.name}: temp={self.last_temp:.1f}"

    def get_status(self, eventtime: float):
        del eventtime
        return {
            "temperature": round(self.last_temp, 2),
            "measured_min_temp": round(self.measured_min, 2),
            "measured_max_temp": round(self.measured_max, 2),
        }
