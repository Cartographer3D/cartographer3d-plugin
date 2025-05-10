from __future__ import annotations

import logging
from textwrap import dedent
from typing import TYPE_CHECKING, Callable, TypedDict, final

from cartographer.adapters.shared.axis_twist_compensation import KlipperAxisTwistCompensationHelper
from cartographer.adapters.shared.bed_mesh import KlipperMeshHelper
from cartographer.adapters.shared.configuration import KlipperConfiguration
from cartographer.adapters.shared.endstop import KlipperEndstop
from cartographer.adapters.shared.homing import CartographerHomingChip
from cartographer.adapters.shared.logging import setup_console_logger
from cartographer.adapters.shared.mcu import KlipperCartographerMcu
from cartographer.adapters.shared.mcu.mcu import Sample
from cartographer.adapters.shared.printer import KlipperToolhead
from cartographer.adapters.shared.probe import KlipperCartographerProbe
from cartographer.adapters.shared.task_executor import KlipperMultiprocessingExecutor
from cartographer.adapters.shared.temperature import PrinterTemperatureCoil
from cartographer.lib.alpha_beta_filter import AlphaBetaFilter
from cartographer.macros import ProbeAccuracyMacro, ProbeMacro, QueryProbeMacro, ZOffsetApplyProbeMacro
from cartographer.macros.axis_twist_compensation import AxisTwistCompensationMacro
from cartographer.macros.backlash import EstimateBacklashMacro
from cartographer.macros.bed_mesh import BedMeshCalibrateMacro
from cartographer.macros.scan_calibrate import ScanCalibrateMacro
from cartographer.macros.touch import TouchAccuracyMacro, TouchCalibrateMacro, TouchHomeMacro, TouchMacro
from cartographer.probe import Probe, ScanMode, ScanModel, TouchMode

if TYPE_CHECKING:
    from configfile import ConfigWrapper
    from gcode import GCodeCommand

    from cartographer.interfaces.printer import Macro

logger = logging.getLogger(__name__)


def load_config(config: ConfigWrapper):
    pheaters = config.get_printer().load_object(config, "heaters")
    pheaters.add_sensor_factory("cartographer_coil", PrinterTemperatureCoil)
    return PrinterCartographer(config)


def smooth_with(filter: AlphaBetaFilter) -> Callable[[Sample], Sample]:
    def fn(sample: Sample) -> Sample:
        return Sample(
            sample.time,
            filter.update(measurement=sample.frequency, time=sample.time),
            sample.temperature,
            sample.position,
            sample.velocity,
        )

    return fn


class CartographerStatus(TypedDict):
    scan: CartographerScanStatus
    touch: CartographerTouchStatus


class CartographerScanStatus(TypedDict):
    current_model: str | None
    last_z_result: float | None


class CartographerTouchStatus(TypedDict):
    current_model: str | None
    last_z_result: float | None


@final
class PrinterCartographer:
    config: KlipperConfiguration

    def __init__(self, config: ConfigWrapper) -> None:
        printer = config.get_printer()
        logger.debug("Initializing Cartographer")
        self.config = KlipperConfiguration(config)
        task_executor = KlipperMultiprocessingExecutor(printer.get_reactor())

        filter = AlphaBetaFilter()
        self.mcu = KlipperCartographerMcu(config, smooth_with(filter))
        toolhead = KlipperToolhead(config, self.mcu)

        self.scan_mode = ScanMode(self.mcu, toolhead, self.config)
        scan_endstop = KlipperEndstop(self.mcu, self.scan_mode)

        self.touch_mode = TouchMode(self.mcu, toolhead, self.config)
        probe = Probe(self.scan_mode, self.touch_mode)

        homing_chip = CartographerHomingChip(printer, scan_endstop)

        printer.lookup_object("pins").register_chip("probe", homing_chip)

        self.gcode = printer.lookup_object("gcode")
        self._configure_macro_logger()
        self.probe_macro = ProbeMacro(probe)
        self._register_macro(self.probe_macro)
        self._register_macro(ProbeAccuracyMacro(probe, toolhead))
        query_probe_macro = QueryProbeMacro(probe)
        self._register_macro(query_probe_macro)

        self._register_macro(ZOffsetApplyProbeMacro(probe, toolhead))

        self.touch_macro = TouchMacro(self.touch_mode)
        self._register_macro(self.touch_macro)
        self._register_macro(TouchAccuracyMacro(self.touch_mode, toolhead))
        touch_home = TouchHomeMacro(self.touch_mode, toolhead, self.config.bed_mesh.zero_reference_position)
        self._register_macro(touch_home)

        self._register_macro(
            BedMeshCalibrateMacro(
                probe,
                toolhead,
                KlipperMeshHelper(config, self.gcode),
                task_executor,
                self.config,
            )
        )

        self._register_macro(ScanCalibrateMacro(probe, toolhead, self.config))
        self._register_macro(TouchCalibrateMacro(self.touch_mode, toolhead, self.config))

        self._register_macro(
            AxisTwistCompensationMacro(probe, toolhead, KlipperAxisTwistCompensationHelper(config), self.config)
        )
        self._register_macro(EstimateBacklashMacro(toolhead, self.scan_mode))

        printer.add_object(
            "probe",
            KlipperCartographerProbe(
                toolhead,
                self.scan_mode,
                self.probe_macro,
                query_probe_macro,
            ),
        )

    def _register_macro(self, macro: Macro[GCodeCommand]) -> None:
        self.gcode.register_command(macro.name, catch_macro_errors(macro.run), desc=macro.description)

    def _configure_macro_logger(self) -> None:
        handler = setup_console_logger(self.gcode)

        log_level = logging.DEBUG if self.config.verbose else logging.INFO
        handler.setLevel(log_level)

    def get_status(self, eventtime: float) -> CartographerStatus:
        del eventtime
        return CartographerStatus(
            scan=CartographerScanStatus(
                current_model=self.scan_mode.model.name if self.scan_mode.model else None,
                last_z_result=self.probe_macro.last_trigger_position,
            ),
            touch=CartographerTouchStatus(
                current_model=self.touch_mode.model.name if self.touch_mode.model else None,
                last_z_result=self.touch_macro.last_trigger_position,
            ),
        )


def catch_macro_errors(func: Callable[[GCodeCommand], None]) -> Callable[[GCodeCommand], None]:
    def wrapper(gcmd: GCodeCommand) -> None:
        try:
            return func(gcmd)
        except RuntimeError as e:
            raise gcmd.error(dedent(str(e)).replace("\n", " ").strip()) from e

    return wrapper
