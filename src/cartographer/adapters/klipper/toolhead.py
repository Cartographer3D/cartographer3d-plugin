from __future__ import annotations

from typing import TYPE_CHECKING, cast

from typing_extensions import override

from cartographer.adapters.klipper_like.toolhead import KlipperLikeToolhead

if TYPE_CHECKING:
    from configfile import ConfigWrapper
    from gcode import GCodeCommand, GCodeDispatch

    from cartographer.interfaces.printer import MacroParams
    from cartographer.mcu.mcu import CartographerMcu


class KlipperToolhead(KlipperLikeToolhead):
    def __init__(self, config: ConfigWrapper, mcu: CartographerMcu) -> None:
        super().__init__(config, mcu)
        self._gcode: GCodeDispatch = config.printer.lookup_object("gcode")

    @override
    def get_max_accel(self) -> float:
        return self.toolhead.get_max_velocity()[1]

    @override
    def set_max_accel(self, accel: float, *, gcmd: MacroParams) -> None:
        script = f"SET_VELOCITY_LIMIT ACCEL={accel:.3f}"
        raw_gcmd = cast("GCodeCommand", gcmd)
        if callable(getattr(raw_gcmd, "run_script", None)):
            raw_gcmd.run_script(script)
        else:
            self._gcode.run_script_from_command(script)
