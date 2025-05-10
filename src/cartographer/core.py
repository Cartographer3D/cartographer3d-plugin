from __future__ import annotations

import logging
from typing import TYPE_CHECKING, final

from cartographer.macros.axis_twist_compensation import AxisTwistCompensationMacro
from cartographer.macros.backlash import EstimateBacklashMacro
from cartographer.macros.probe import ProbeAccuracyMacro, ProbeMacro, QueryProbeMacro, ZOffsetApplyProbeMacro
from cartographer.macros.scan_calibrate import ScanCalibrateMacro
from cartographer.macros.touch import TouchAccuracyMacro, TouchHomeMacro, TouchMacro
from cartographer.macros.touch_calibrate import TouchCalibrateMacro
from cartographer.probe.probe import Probe
from cartographer.probe.scan_mode import ScanMode, ScanModeConfiguration
from cartographer.probe.touch_mode import TouchMode, TouchModeConfiguration

if TYPE_CHECKING:
    from cartographer.interfaces.printer import Macro
    from cartographer.runtime.adapters import Adapters

logger = logging.getLogger(__name__)


@final
class PrinterCartographer:
    def __init__(self, adapters: Adapters) -> None:
        mcu = adapters.mcu
        toolhead = adapters.toolhead
        config = adapters.config

        self.scan_mode = ScanMode(
            mcu,
            toolhead,
            ScanModeConfiguration.from_config(config),
        )
        self.touch_mode = TouchMode(mcu, toolhead, TouchModeConfiguration.from_config(config))
        probe = Probe(self.scan_mode, self.touch_mode)

        self.macros: list[Macro] = [
            ProbeMacro(probe),
            ProbeAccuracyMacro(probe, toolhead),
            QueryProbeMacro(probe),
            ZOffsetApplyProbeMacro(probe, toolhead, config),
            TouchCalibrateMacro(probe, mcu, toolhead, config),
            TouchMacro(self.touch_mode),
            TouchAccuracyMacro(self.touch_mode, toolhead),
            TouchHomeMacro(self.touch_mode, toolhead, config.bed_mesh.zero_reference_position),
            # BedMeshCalibrateMacro(...),  # Pass dependencies as needed
            ScanCalibrateMacro(probe, toolhead, config),
            EstimateBacklashMacro(toolhead, self.scan_mode),
        ]

        if adapters.axis_twist_compensation:
            self.macros.append(AxisTwistCompensationMacro(probe, toolhead, adapters.axis_twist_compensation, config))

    def get_status(self, eventtime: float) -> object:
        return {
            "scan": self.scan_mode.get_status(eventtime),
            "touch": self.touch_mode.get_status(eventtime),
        }
