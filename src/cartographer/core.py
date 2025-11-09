from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING, final

from cartographer.coil.temperature_compensation import CoilTemperatureCompensationModel
from cartographer.macros.axis_twist_compensation import AxisTwistCompensationMacro
from cartographer.macros.backlash import EstimateBacklashMacro
from cartographer.macros.bed_mesh.scan_mesh import BedMeshCalibrateConfiguration, BedMeshCalibrateMacro
from cartographer.macros.migration_message import MigrationMessageMacro
from cartographer.macros.model_manager import ScanModelManager, TouchModelManager
from cartographer.macros.probe import ProbeAccuracyMacro, ProbeMacro, QueryProbeMacro, ZOffsetApplyProbeMacro
from cartographer.macros.query import QueryMacro
from cartographer.macros.scan import ScanAccuracyMacro
from cartographer.macros.scan_calibrate import DEFAULT_SCAN_MODEL_NAME, ScanCalibrateMacro
from cartographer.macros.stream import StreamMacro
from cartographer.macros.temperature_calibrate import TemperatureCalibrateMacro
from cartographer.macros.touch import (
    DEFAULT_TOUCH_MODEL_NAME,
    TouchAccuracyMacro,
    TouchCalibrateMacro,
    TouchHomeMacro,
    TouchProbeMacro,
)
from cartographer.probe.probe import Probe
from cartographer.probe.scan_mode import ScanMode, ScanModeConfiguration
from cartographer.probe.touch_mode import TouchMode, TouchModeConfiguration
from cartographer.toolhead import BacklashCompensatingToolhead

if TYPE_CHECKING:
    from cartographer.interfaces.printer import Macro
    from cartographer.runtime.adapters import Adapters

logger = logging.getLogger(__name__)


@dataclass
class MacroRegistration:
    name: str
    macro: Macro


@final
class PrinterCartographer:
    def __init__(self, adapters: Adapters) -> None:
        self.mcu = adapters.mcu
        self.config = adapters.config
        toolhead = (
            BacklashCompensatingToolhead(adapters.toolhead, self.config.general.z_backlash)
            if self.config.general.z_backlash > 0
            else adapters.toolhead
        )

        self.scan_mode = ScanMode(
            self.mcu,
            toolhead,
            ScanModeConfiguration.from_config(self.config),
            CoilTemperatureCompensationModel(self.config.coil.calibration, adapters.mcu)
            if self.config.coil.calibration
            else None,
            adapters.axis_twist_compensation,
        )
        if DEFAULT_SCAN_MODEL_NAME in adapters.config.scan.models:
            self.scan_mode.load_model(DEFAULT_SCAN_MODEL_NAME)

        self.touch_mode = TouchMode(self.mcu, toolhead, TouchModeConfiguration.from_config(self.config))
        if DEFAULT_TOUCH_MODEL_NAME in adapters.config.touch.models:
            self.touch_mode.load_model(DEFAULT_TOUCH_MODEL_NAME)

        probe = Probe(self.scan_mode, self.touch_mode)

        def reg(name: str, macro: Macro, use_prefix: bool = True) -> list[MacroRegistration]:
            if not use_prefix:
                return [MacroRegistration(name, macro)]

            registrations = [MacroRegistration(f"CARTOGRAPHER_{name}", macro)]

            prefix = self.config.general.macro_prefix
            if prefix is not None:
                formatted_prefix = prefix.rstrip("_").upper() + "_" if prefix else ""
                registrations.append(MacroRegistration(f"{formatted_prefix}{name}", macro))

            return registrations

        self.probe_macro = ProbeMacro(probe)
        self.query_probe_macro = QueryProbeMacro(probe)
        self.macros = list(
            chain.from_iterable(
                [
                    reg("PROBE", self.probe_macro, use_prefix=False),
                    reg("PROBE_ACCURACY", ProbeAccuracyMacro(probe, toolhead), use_prefix=False),
                    reg("QUERY_PROBE", self.query_probe_macro, use_prefix=False),
                    reg("Z_OFFSET_APPLY_PROBE", ZOffsetApplyProbeMacro(probe, toolhead, self.config), use_prefix=False),
                    reg("QUERY", QueryMacro(self.mcu, self.scan_mode, self.touch_mode)),
                    reg(
                        "BED_MESH_CALIBRATE",
                        BedMeshCalibrateMacro(
                            probe,
                            toolhead,
                            adapters.bed_mesh,
                            adapters.axis_twist_compensation,
                            adapters.task_executor,
                            BedMeshCalibrateConfiguration.from_config(self.config),
                        ),
                        use_prefix=False,
                    ),
                    reg("STREAM", StreamMacro(self.mcu)),
                    reg(
                        "TEMPERATURE_CALIBRATE",
                        TemperatureCalibrateMacro(
                            self.mcu, toolhead, self.config, adapters.gcode, adapters.task_executor
                        ),
                    ),
                    reg("SCAN_CALIBRATE", ScanCalibrateMacro(probe, toolhead, self.config)),
                    reg("SCAN_ACCURACY", ScanAccuracyMacro(self.scan_mode, toolhead, self.mcu)),
                    reg("SCAN_MODEL", ScanModelManager(self.scan_mode, self.config)),
                    reg("ESTIMATE_BACKLASH", EstimateBacklashMacro(toolhead, self.scan_mode, self.config)),
                    reg("TOUCH_CALIBRATE", TouchCalibrateMacro(probe, self.mcu, toolhead, self.config)),
                    reg("TOUCH_MODEL", TouchModelManager(self.touch_mode, self.config)),
                    reg("TOUCH_PROBE", TouchProbeMacro(self.touch_mode)),
                    reg(
                        "TOUCH_ACCURACY",
                        TouchAccuracyMacro(self.touch_mode, toolhead, lift_speed=self.config.general.lift_speed),
                    ),
                    reg(
                        "TOUCH_HOME",
                        TouchHomeMacro(
                            self.touch_mode,
                            toolhead,
                            lift_speed=self.config.general.lift_speed,
                            home_position=self.config.bed_mesh.zero_reference_position,
                            travel_speed=self.config.general.travel_speed,
                            random_radius=self.config.touch.home_random_radius,
                        ),
                    ),
                ]
            )
        )

        if adapters.axis_twist_compensation:
            self.macros.extend(
                reg(
                    "CARTOGRAPHER_AXIS_TWIST_COMPENSATION",
                    AxisTwistCompensationMacro(probe, toolhead, adapters.axis_twist_compensation, self.config),
                    use_prefix=False,
                )
            )

        old_macros = list(
            chain.from_iterable(
                [
                    reg("TOUCH", MigrationMessageMacro("CARTOGRAPHER_TOUCH", "CARTOGRAPHER_TOUCH_HOME")),
                    reg("CALIBRATE", MigrationMessageMacro("CARTOGRAPHER_CALIBRATE", "CARTOGRAPHER_SCAN_CALIBRATE")),
                    reg(
                        "THRESHOLD_SCAN",
                        MigrationMessageMacro("CARTOGRAPHER_THRESHOLD_SCAN", "CARTOGRAPHER_TOUCH_CALIBRATE"),
                    ),
                ]
            )
        )
        self.macros.extend(old_macros)

    def get_status(self, eventtime: float) -> object:
        return {
            "scan": self.scan_mode.get_status(eventtime),
            "touch": self.touch_mode.get_status(eventtime),
            "mcu": self.mcu.get_status(eventtime),
        }
