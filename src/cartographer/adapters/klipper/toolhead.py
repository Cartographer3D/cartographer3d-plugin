from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import override

from cartographer.adapters.klipper_like.gcode_context import get_gcode_command
from cartographer.adapters.klipper_like.toolhead import KlipperLikeToolhead

if TYPE_CHECKING:
    from configfile import ConfigWrapper
    from gcode import GCodeDispatch

    from cartographer.mcu.mcu import CartographerMcu


class KlipperToolhead(KlipperLikeToolhead):
    def __init__(self, config: ConfigWrapper, mcu: CartographerMcu) -> None:
        super().__init__(config, mcu)
        self._gcode: GCodeDispatch = config.printer.lookup_object("gcode")

    @override
    def get_max_accel(self) -> float:
        return self.toolhead.get_max_velocity()[1]

    @override
    def set_max_accel(self, accel: float) -> None:
        script = f"SET_VELOCITY_LIMIT ACCEL={accel:.3f}"
        gcmd = get_gcode_command()
        if gcmd is not None and callable(getattr(gcmd, "run_script", None)):
            gcmd.run_script(script)
        else:
            self._gcode.run_script_from_command(script)
