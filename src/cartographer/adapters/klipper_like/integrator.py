from __future__ import annotations

import logging
from abc import ABC
from functools import wraps
from textwrap import dedent
from typing import TYPE_CHECKING, Callable, Protocol, Sequence, final

from gcode import GCodeCommand, GCodeDispatch
from typing_extensions import override

from cartographer.adapters.klipper.endstop import KlipperEndstop, KlipperHomingState
from cartographer.adapters.klipper.homing import KlipperHomingChip
from cartographer.adapters.klipper.logging import setup_console_logger
from cartographer.adapters.klipper.temperature import PrinterTemperatureCoil
from cartographer.adapters.klipper_like.utils import reraise_for_klipper
from cartographer.interfaces.printer import Macro, MacroParams, SupportsFallbackMacro
from cartographer.runtime.integrator import Integrator

if TYPE_CHECKING:
    from extras.homing import Homing
    from klippy import Printer
    from mcu import MCU_endstop
    from stepper import MCU_stepper

    from cartographer.adapters.klipper.configuration import KlipperConfiguration
    from cartographer.adapters.klipper.mcu.mcu import KlipperCartographerMcu
    from cartographer.core import MacroRegistration
    from cartographer.interfaces.printer import Endstop

logger = logging.getLogger(__name__)


class KlipperLikeAdapters(Protocol):
    mcu: KlipperCartographerMcu
    printer: Printer
    config: KlipperConfiguration


class _Rail(Protocol):
    def get_steppers(self) -> list[MCU_stepper]: ...
    def get_endstops(self) -> list[tuple[MCU_endstop, str]]: ...


class KlipperLikeIntegrator(Integrator, ABC):
    def __init__(self, adapters: KlipperLikeAdapters) -> None:
        self._config: KlipperConfiguration = adapters.config
        self._printer: Printer = adapters.printer
        self._mcu: KlipperCartographerMcu = adapters.mcu

        self._gcode: GCodeDispatch = self._printer.lookup_object("gcode")

    @override
    def setup(self) -> None:
        self._printer.register_event_handler("homing:home_rails_end", self._handle_home_rails_end)
        self._configure_macro_logger()

    @override
    def register_endstop_pin(self, chip_name: str, pin: str, endstop: Endstop) -> None:
        mcu_endstop = KlipperEndstop(self._mcu, endstop)
        chip = KlipperHomingChip(mcu_endstop, pin)
        self._printer.lookup_object("pins").register_chip(chip_name, chip)

    @override
    def register_macro(self, registration: MacroRegistration) -> None:
        name = registration.name
        macro = registration.macro
        if isinstance(macro, SupportsFallbackMacro):
            original = self._gcode.register_command(name, None)
            if original:
                macro.set_fallback_macro(FallbackMacroAdapter(name, original))
            else:
                logger.warning("No original macro found to fallback to for '%s'", name)

        self._gcode.register_command(name, _catch_macro_errors(macro.run), desc=macro.description)

    @override
    def register_coil_temperature_sensor(self) -> None:
        pheaters = self._printer.load_object(self._config.wrapper, "heaters")
        sensor = PrinterTemperatureCoil(self._mcu, self._config.coil)

        object_name = f"temperature_sensor {sensor.name}"
        self._printer.add_object(object_name, sensor)
        pheaters.available_sensors.append(object_name)

    @override
    def register_ready_callback(self, callback: Callable[[], None]) -> None:
        self._printer.register_event_handler("klippy:ready", callback)

    @reraise_for_klipper
    def _handle_home_rails_end(self, homing: Homing, rails: Sequence[_Rail]) -> None:
        homing_state = KlipperHomingState(homing)
        klipper_endstops = [
            es.endstop for rail in rails for es, _ in rail.get_endstops() if isinstance(es, KlipperEndstop)
        ]
        for endstop in klipper_endstops:
            endstop.on_home_end(homing_state)

    def _configure_macro_logger(self) -> None:
        handler = setup_console_logger(self._gcode)
        log_level = logging.DEBUG if self._config.general.verbose else logging.INFO
        handler.setLevel(log_level)


def _catch_macro_errors(func: Callable[[GCodeCommand], None]) -> Callable[[GCodeCommand], None]:
    @wraps(func)
    def wrapper(gcmd: GCodeCommand) -> None:
        try:
            func(gcmd)
        except (RuntimeError, ValueError) as e:
            msg = dedent(str(e)).replace("\n", " ").replace("  ", "\n").strip()
            raise gcmd.error(msg) from e

    return wrapper


@final
class FallbackMacroAdapter(Macro):
    def __init__(self, name: str, handler: Callable[[GCodeCommand], None]) -> None:
        self.name = name
        self.description: str = f"Fallback for {name}"
        self._handler: Callable[[GCodeCommand], None] = handler

    @override
    def run(self, params: MacroParams) -> None:
        assert isinstance(params, GCodeCommand), f"Invalid gcode params type for {self.name}"
        self._handler(params)
