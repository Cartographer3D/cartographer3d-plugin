from __future__ import annotations

import logging
import struct
from typing import TYPE_CHECKING, TypedDict, final

from extras.thermistor import Thermistor

if TYPE_CHECKING:
    from mcu import MCU, CommandQueryWrapper

logger = logging.getLogger(__name__)


class _BaseData(TypedDict):
    bytes: bytes


TRIGGER_HYSTERESIS = 0.006

SHORTED_FREQUENCY_VALUE = 0xFFFFFFF
FREQUENCY_RANGE_PERCENT = 1.35
UINT32_MAX = 0xFFFFFFFF
UINT16_MAX = 0xFFFF


@final
class KlipperCartographerConstants:
    _sensor_frequency: float = 1
    _inverse_adc_max: float = 0.0
    _adc_smooth_count: int = 1

    minimum_adc_count: int = 0
    minimum_count: int = 0

    def __init__(self, mcu: MCU):
        self._mcu = mcu
        self._command_queue = self._mcu.alloc_command_queue()
        self._mcu.register_config_callback(self._initialize_constants)

        self.thermistor = Thermistor(10000.0, 0.0)
        self.thermistor.setup_coefficients_beta(25.0, 47000.0, 4041.0)

    def _initialize_constants(self):
        constants = self._mcu.get_constants()
        self._sensor_frequency = self._clock_to_sensor_frequency(float(constants["CLOCK_FREQ"]))
        self._inverse_adc_max = 1.0 / int(constants["ADC_MAX"])
        self._adc_smooth_count = int(constants["CARTOGRAPHER_ADC_SMOOTH_COUNT"])
        logger.debug("Received constants: %s", constants)

        base_read_command = self._mcu.lookup_query_command(
            "cartographer_base_read len=%c offset=%hu",
            "cartographer_base_data bytes=%*s offset=%hu",
            cq=self._command_queue,
        )
        self._read_base(base_read_command)

    def _read_base(self, cmd: CommandQueryWrapper[_BaseData]) -> None:
        fixed_length = 6
        fixed_offset = 0

        base_data = cmd.send([fixed_length, fixed_offset])

        f_count: int
        adc_count: int
        f_count, adc_count = struct.unpack("<IH", base_data["bytes"])

        if f_count >= UINT32_MAX or adc_count >= UINT16_MAX:
            msg = "invalid f_count or adc_count"
            raise self._mcu.error(msg)

        self.minimum_adc_count = adc_count
        self.minimum_count = f_count

    def _clock_to_sensor_frequency(self, clock_frequency: float) -> float:
        if clock_frequency < 20e6:  # noqa: PLR2004
            return clock_frequency
        if clock_frequency < 100e6:  # noqa: PLR2004
            return clock_frequency / 2
        return clock_frequency / 6

    def count_to_frequency(self, count: int) -> float:
        return count * self._sensor_frequency / (2**28)

    def frequency_to_count(self, frequency: float) -> int:
        return int(frequency * (2**28) / self._sensor_frequency)

    def calculate_temperature(self, raw_temp: int) -> float:
        temp_adc = raw_temp / self._adc_smooth_count * self._inverse_adc_max
        return self.thermistor.calc_temp(temp_adc)
