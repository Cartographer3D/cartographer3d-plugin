from typing import TYPE_CHECKING

from configfile import ConfigWrapper
from typing_extensions import override

from cartographer.adapters.klipper.mcu.mcu import KlipperCartographerMcu
from cartographer.adapters.klipper_like.toolhead import KlipperLikeToolhead

if TYPE_CHECKING:
    from gcode import GCodeDispatch


class KlipperToolhead(KlipperLikeToolhead):
    def __init__(self, config: ConfigWrapper, mcu: KlipperCartographerMcu) -> None:
        super().__init__(config, mcu)
        self._gcode: GCodeDispatch = config.printer.lookup_object("gcode")

    @override
    def get_max_accel(self) -> float:
        return self.toolhead.get_max_velocity()[1]

    @override
    def set_max_accel(self, accel: float) -> None:
        self._gcode.run_script_from_command(f"SET_VELOCITY_LIMIT ACCEL={accel:.3f}")
