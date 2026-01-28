from typing_extensions import override

from cartographer.adapters.klipper_like.axis_twist_compensation import KlipperLikeAxisTwistCompensationAdapter


class KlipperAxisTwistCompensationAdapter(KlipperLikeAxisTwistCompensationAdapter):
    @override
    def get_z_compensation_value(self, *, x: float, y: float) -> float:
        # Import manual_probe module to create ProbeResult objects
        import importlib
        manual_probe = importlib.import_module(".manual_probe", "extras")
        # Create a ProbeResult object instead of a plain list for compatibility with axis_twist_compensation
        probe_result = manual_probe.ProbeResult(x, y, 0.0, x, y, 0.0)
        pos_list = [probe_result]
        self.printer.send_event("probe:update_results", pos_list)
        # Return the z compensation from the modified probe result
        return pos_list[0].bed_z

