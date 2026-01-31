from __future__ import annotations

import logging
from typing import TYPE_CHECKING, final

from typing_extensions import override

from cartographer.interfaces.printer import Macro, MacroParams, Position, Toolhead

if TYPE_CHECKING:
    from cartographer.probe.touch_mode import TouchMode


logger = logging.getLogger(__name__)


@final
class TouchProbeMacro(Macro):
    description = "Touch the bed to get the height offset at the current position."
    last_trigger_position: float | None = None
    last_probe_position: Position | None = None

    def __init__(self, probe: TouchMode, toolhead: Toolhead) -> None:
        self._probe = probe
        self._toolhead = toolhead

    @override
    def run(self, params: MacroParams) -> None:
        trigger_pos = self._probe.perform_probe()
        self.last_trigger_position = trigger_pos
        offset = self._probe.offset
        pos = self._toolhead.get_position()
        self.last_probe_position = Position(pos.x + offset.x, pos.y + offset.y, z=trigger_pos - offset.z)
        logger.info(
            "Result: at %.3f,%.3f estimate contact at z=%.6f",
            self.last_probe_position.x,
            self.last_probe_position.y,
            self.last_probe_position.z,
        )
