from __future__ import annotations

import logging
from typing import TYPE_CHECKING, final

from cartographer.macros.backlash import EstimateBacklashMacro
from cartographer.macros.probe import ProbeAccuracyMacro, ProbeMacro, QueryProbeMacro, ZOffsetApplyProbeMacro
from cartographer.macros.scan_calibrate import ScanCalibrateMacro
from cartographer.macros.touch import TouchAccuracyMacro, TouchHomeMacro, TouchMacro
from cartographer.macros.touch_calibrate import TouchCalibrateMacro
from cartographer.probe.probe import Probe
from cartographer.probe.scan_mode import ScanMode, ScanModeConfiguration
from cartographer.probe.touch_mode import TouchMode, TouchModeConfiguration

if TYPE_CHECKING:
    from cartographer.runtime.types import Adapters

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
        touch_mode = TouchMode(mcu, toolhead, TouchModeConfiguration.from_config(config))
        probe = Probe(self.scan_mode, touch_mode)

        self.macros = [
            ProbeMacro(probe),
            ProbeAccuracyMacro(probe, toolhead),
            QueryProbeMacro(probe),
            ZOffsetApplyProbeMacro(probe, toolhead, config),
            TouchCalibrateMacro(probe, mcu, toolhead, config),
            TouchMacro(touch_mode),
            TouchAccuracyMacro(touch_mode, toolhead),
            TouchHomeMacro(touch_mode, toolhead, config.bed_mesh.zero_reference_position),
            # BedMeshCalibrateMacro(...),  # Pass dependencies as needed
            ScanCalibrateMacro(probe, toolhead, config),
            # AxisTwistCompensationMacro(...),
            EstimateBacklashMacro(toolhead, self.scan_mode),
        ]

    # TODO: Move this into an adapter
    def get_status(self, eventtime: float) -> object:
        del eventtime
        return {
            "scan": {},
            "touch": {},
        }
