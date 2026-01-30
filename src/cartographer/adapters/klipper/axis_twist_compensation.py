from extras import manual_probe
from typing_extensions import override

from cartographer.adapters.klipper_like.axis_twist_compensation import KlipperLikeAxisTwistCompensationAdapter


class KlipperAxisTwistCompensationAdapter(KlipperLikeAxisTwistCompensationAdapter):
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
        probe_result = manual_probe.ProbeResult(x, y, 0.0, x, y, 0.0)
        pos_list = [probe_result]
        self.printer.send_event("probe:update_results", pos_list)
        return pos_list[0].bed_z
