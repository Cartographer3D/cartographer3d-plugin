from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from typing_extensions import override

from cartographer.macros.axis_twist_compensation import (
    AxisTwistCompensationAdapter,
    CalibrationOptions,
    CompensationResult,
)

if TYPE_CHECKING:
    from configfile import ConfigWrapper, PrinterConfig
    from extras.axis_twist_compensation import AxisTwistCompensation as KlipperAxisTwistCompensation
    from klippy import Printer


class KlipperV12AxisTwistCompensationAdapter(AxisTwistCompensationAdapter):
    def __init__(self, config: ConfigWrapper) -> None:
        self.config: ConfigWrapper = config.getsection("axis_twist_compensation")
        self.printer: Printer = config.get_printer()
        self.compensation: KlipperAxisTwistCompensation = self.printer.load_object(
            self.config, "axis_twist_compensation"
        )
        self.configfile: PrinterConfig = self.printer.lookup_object("configfile")
        self.configname: str = self.config.get_name()

        self.move_height: float = self.compensation.horizontal_move_z
        self.speed: float = self.compensation.speed
        self.available_axes: tuple[Literal["x", "y"], ...] = ("x",)

    @override
    def clear_compensations(self, axis: Literal["x", "y"]) -> None:
        if axis == "y":
            msg = "Y-axis twist compensation is not supported on Klipper v0.12"
            raise RuntimeError(msg)
        # v0.12 clear_compensations() takes no arguments
        self.compensation.clear_compensations()

    @override
    def apply_compensation(self, result: CompensationResult) -> None:
        if result.axis == "y":
            msg = "Y-axis twist compensation is not supported on Klipper v0.12"
            raise RuntimeError(msg)

        values_str = ", ".join(f"{v:.6f}" for v in result.values)
        self.configfile.set(self.configname, "z_compensations", values_str)
        self.configfile.set(self.configname, "compensation_start_x", result.start)
        self.configfile.set(self.configname, "compensation_end_x", result.end)

        self.compensation.z_compensations = result.values
        self.compensation.compensation_start_x = result.start
        self.compensation.compensation_end_x = result.end

    @override
    def get_calibration_options(self, axis: Literal["x", "y"]) -> CalibrationOptions:
        if axis == "y":
            msg = "Y-axis twist compensation is not supported on Klipper v0.12"
            raise RuntimeError(msg)
        return CalibrationOptions(
            self.compensation.calibrate_start_x,
            self.compensation.calibrate_end_x,
            self.compensation.calibrate_y,
        )

    @override
    def get_z_compensation_value(self, *, x: float, y: float) -> float:
        # v0.12 get_z_compensation_value takes a plain list [x, y, z]
        pos = [x, y, 0]
        self.printer.send_event("probe:update_results", pos)
        return pos[2]
