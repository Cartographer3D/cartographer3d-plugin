from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import override

from cartographer.adapters.klipper.gcode import KlipperGCodeDispatch
from cartographer.adapters.klipper_like.toolhead import KlipperLikeToolhead

if TYPE_CHECKING:
    from configfile import ConfigWrapper

    from cartographer.interfaces.printer import MacroParams
    from cartographer.mcu.mcu import CartographerMcu


class KlipperToolhead(KlipperLikeToolhead):
    def __init__(self, config: ConfigWrapper, mcu: CartographerMcu) -> None:
        super().__init__(config, mcu)
        self._gcode: KlipperGCodeDispatch = KlipperGCodeDispatch(self.printer)

    @override
    def get_max_accel(self) -> float:
        return self.toolhead.get_max_velocity()[1]

    @override
    def set_max_accel(self, accel: float, *, gcmd: MacroParams) -> None:
        self._gcode.run_gcode(f"SET_VELOCITY_LIMIT ACCEL={accel:.3f}", gcmd=gcmd)
