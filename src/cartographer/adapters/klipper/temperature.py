from __future__ import annotations

import logging
from typing import TYPE_CHECKING, final

from cartographer.adapters.klipper.mcu import KlipperCartographerMcu

if TYPE_CHECKING:
    from configfile import ConfigWrapper

    from cartographer.interfaces.printer import Sample

REPORT_TIME = 0.300

logger = logging.getLogger(__name__)


@final
class PrinterTemperatureCoil:
    def __init__(self, config: ConfigWrapper):
        self.printer = config.get_printer()
        self.name = config.get("name", default="cartographer_coil")

        self.min_temp = config.getfloat("min_temp", default=-300, minval=-300)
        self.max_temp = config.getfloat("max_temp", 9999, above=self.min_temp)
        self.printer.register_event_handler("klippy:mcu_identify", self._handle_mcu_identify)

        self.last_temp = 0.0
        self.measured_min = 99999.0
        self.measured_max = -300

    def _handle_mcu_identify(self) -> None:
        carto = self.printer.lookup_object("cartographer")
        if not isinstance(carto.mcu, KlipperCartographerMcu):
            logger.error("Expected cartographer MCU to be of type KlipperCartographerMcu, got %s", type(carto.mcu))
            return
        carto.mcu.register_callback(self._sample_callback)

    def temperature_callback(self, temp: float):
        self.last_temp = temp
        if temp:
            self.measured_min = min(self.measured_min, temp)
            self.measured_max = max(self.measured_max, temp)

    def get_report_time_delta(self) -> float:
        return REPORT_TIME

    def _sample_callback(self, sample: Sample) -> None:
        self.temperature_callback(sample.temperature)
        if not (self.min_temp <= sample.temperature <= self.max_temp):
            logger.warning(
                "temperature for %(sensor_name)s at %(temperature)s is out of range [%(min_temp)s, %(max_temp)s]",
                dict(
                    sensor_name=self.name,
                    temperature=sample.temperature,
                    min_temp=self.min_temp,
                    max_temp=self.max_temp,
                ),
            )
        return

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
