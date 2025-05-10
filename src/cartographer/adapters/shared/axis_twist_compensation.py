from __future__ import annotations

from typing import TYPE_CHECKING, Literal, final

from typing_extensions import override

from cartographer.macros.axis_twist_compensation import (
    AxisTwistCompensationHelper,
    CalibrationOptions,
    CompensationResult,
)

if TYPE_CHECKING:
    from configfile import ConfigWrapper


@final
class KlipperAxisTwistCompensationHelper(AxisTwistCompensationHelper):
    def __init__(self, config: ConfigWrapper) -> None:
        self.config = config.getsection("axis_twist_compensation")
        self.compensation = config.get_printer().load_object(self.config, "axis_twist_compensation")
        self.configfile = config.get_printer().lookup_object("configfile")
        self.configname = self.config.get_name()

        self.move_height = self.compensation.horizontal_move_z
        self.speed = self.compensation.speed

    @override
    def clear_compensations(self, axis: Literal["x", "y"]) -> None:
        self.compensation.clear_compensations(axis.upper())

    @override
    def apply_compensation(self, result: CompensationResult) -> None:
        values_str = ", ".join(f"{v:.6f}" for v in result.values)

        if result.axis == "x":
            self._set_config_value("z_compensations", values_str)
            self._set_config_value("compensation_start_x", result.start)
            self._set_config_value("compensation_end_x", result.end)

            self.compensation.z_compensations = result.values
            self.compensation.compensation_start_x = result.start
            self.compensation.compensation_end_x = result.end

        elif result.axis == "y":
            self._set_config_value("zy_compensations", values_str)
            self._set_config_value("compensation_start_y", result.start)
            self._set_config_value("compensation_end_y", result.end)

            self.compensation.zy_compensations = result.values
            self.compensation.compensation_start_y = result.start
            self.compensation.compensation_end_y = result.end

    def _set_config_value(self, key: str, value: float | str) -> None:
        self.configfile.set(self.configname, key, value)

    @override
    def get_calibration_options(self, axis: Literal["x", "y"]) -> CalibrationOptions:
        if axis == "x":
            return CalibrationOptions(
                self.compensation.calibrate_start_x,
                self.compensation.calibrate_end_x,
                self.compensation.calibrate_y,
            )
        else:
            return CalibrationOptions(
                self.compensation.calibrate_start_y,
                self.compensation.calibrate_end_y,
                self.compensation.calibrate_x,
            )
