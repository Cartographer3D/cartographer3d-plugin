from __future__ import annotations

from typing import TYPE_CHECKING

from extras import manual_probe
from typing_extensions import override

from cartographer.adapters.klipper_like.axis_twist_compensation import KlipperLikeAxisTwistCompensationAdapter

if TYPE_CHECKING:
    from configfile import ConfigWrapper


class KlipperAxisTwistCompensationAdapter(KlipperLikeAxisTwistCompensationAdapter):
    def __init__(self, config: ConfigWrapper, *, x_offset: float, y_offset: float) -> None:
        super().__init__(config)
        self._x_offset: float = x_offset
        self._y_offset: float = y_offset

    @override
    def get_z_compensation_value(self, *, x: float, y: float) -> float:
        # Support both old and new Klipper versions
        # Check if ProbeResult exists (Klipper >= v0.13.0-465, introduced Dec 2025)
        if not hasattr(manual_probe, "ProbeResult"):
            # Old Klipper: send plain list
            pos = [x, y, 0]
            self.printer.send_event("probe:update_results", pos)
            return pos[2]

        # New Klipper: send ProbeResult object
        # Handlers may use nozzle test coordinates or offset probe bed coordinates.
        probe_result = manual_probe.ProbeResult(x + self._x_offset, y + self._y_offset, 0.0, x, y, 0.0)
        pos_list = [probe_result]
        self.printer.send_event("probe:update_results", pos_list)
        return pos_list[0].bed_z
