from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, final

if TYPE_CHECKING:
    from configfile import ConfigWrapper

    from cartographer.interfaces.printer import Sample

REPORT_TIME = 0.300

logger = logging.getLogger(__name__)


@final
class PrinterTemperatureCoil:
    def __init__(self, config: ConfigWrapper):
        self.printer = config.get_printer()
        self.name = config.get_name()
        self.min_temp = float("-inf")
        self.max_temp = float("inf")
        self.temperature_callback = None
        self.printer.register_event_handler("klippy:mcu_identify", self._handle_mcu_identify)

    def _handle_mcu_identify(self) -> None:
        carto = self.printer.lookup_object("cartographer")
        carto.mcu.register_callback(self._sample_callback)

    def setup_callback(self, temperature_callback: Callable[[float, float], None]) -> None:
        self.temperature_callback = temperature_callback

    def get_report_time_delta(self) -> float:
        return REPORT_TIME

    def setup_minmax(self, min_temp: float, max_temp: float) -> None:
        self.min_temp = min_temp
        self.max_temp = max_temp

    def _sample_callback(self, sample: Sample) -> None:
        if self.temperature_callback is None:
            return
        self.temperature_callback(sample.time, sample.temperature)
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
