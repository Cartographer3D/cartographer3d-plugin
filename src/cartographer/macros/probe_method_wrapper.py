from __future__ import annotations

import logging
from typing import TYPE_CHECKING, final

from typing_extensions import override

from cartographer.interfaces.printer import GCodeDispatch, MacroParams, SupportsFallbackMacro

if TYPE_CHECKING:
    from cartographer.probe import Probe

logger = logging.getLogger(__name__)


@final
class ProbeMethodWrapperMacro(SupportsFallbackMacro):
    description = None

    def __init__(self, probe: Probe, gcode: GCodeDispatch) -> None:
        super().__init__()
        self.probe = probe
        self.gcode = gcode

    @override
    def run(self, params: MacroParams) -> None:
        probe_method = params.get("PROBE_METHOD", "scan").lower()
        if probe_method != "touch":
            self.fallback.run(params)
            return

        # TODO: Avoid overriding these if not BED_MESH_CALIBRATE
        boundaries = self.probe.touch.boundaries
        mesh_min = params.get("MESH_MIN", f"{boundaries.min_x},{boundaries.min_y}")
        mesh_max = params.get("MESH_MAX", f"{boundaries.max_x},{boundaries.max_y}")

        with self.probe.as_touch():
            new_params = self.gcode.clone_params(params, {"MESH_MIN": mesh_min, "MESH_MAX": mesh_max})
            self.fallback.run(new_params)
