# https://github.com/Klipper3d/klipper/blob/master/klippy/extras/heaters.py
from collections.abc import Callable
from typing import Protocol, TypedDict

from configfile import ConfigWrapper

class _Status(TypedDict):
    temperature: float
    target: float
    power: float

class Heater:
    def get_status(self, eventtime: float) -> _Status: ...

class _Sensor(Protocol):
    def setup_callback(self, temperature_callback: Callable[[float, float], None]) -> None: ...
    def get_report_time_delta(self) -> float: ...
    def setup_minmax(self, min_temp: float, max_temp: float) -> None: ...

class PrinterHeaters:
    available_sensors: list[str]
    def add_sensor_factory(self, sensor_type: str, sensor_factory: type[_Sensor]) -> None: ...
    def register_sensor(self, config: ConfigWrapper, psensor: _Sensor) -> None: ...
