from typing_extensions import override

from cartographer.adapters.klipper_like.axis_twist_compensation import KlipperLikeAxisTwistCompensationAdapter


class KlipperAxisTwistCompensationAdapter(KlipperLikeAxisTwistCompensationAdapter):
    @override
    def get_z_compensation_value(self, *, x: float, y: float) -> float:
        # Support both old and new Klipper versions
        import importlib
        try:
            manual_probe = importlib.import_module(".manual_probe", "extras")
            # Check if ProbeResult exists (Klipper >= v0.13.0-465, introduced Dec 2025)
            if hasattr(manual_probe, 'ProbeResult'):
                # New Klipper: send ProbeResult object
                probe_result = manual_probe.ProbeResult(x, y, 0.0, x, y, 0.0)
                pos_list = [probe_result]
                self.printer.send_event("probe:update_results", pos_list)
                return pos_list[0].bed_z
            else:
                # Old Klipper: send plain list
                pos = [x, y, 0]
                self.printer.send_event("probe:update_results", pos)
                return pos[2]
        except (ImportError, AttributeError):
            # Fallback for very old versions
            pos = [x, y, 0]
            self.printer.send_event("probe:update_results", pos)
            return pos[2]

