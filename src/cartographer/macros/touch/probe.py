from __future__ import annotations

import logging
from typing import TYPE_CHECKING, final

from typing_extensions import override

from cartographer.interfaces.printer import Macro, MacroParams

if TYPE_CHECKING:
    from cartographer.probe.touch_mode import TouchMode


logger = logging.getLogger(__name__)


@final
class TouchProbeMacro(Macro):
    description = "Touch the bed to get the height offset at the current position."
    last_trigger_position: float | None = None

    def __init__(self, probe: TouchMode) -> None:
        self._probe = probe

    @override
    def run(self, params: MacroParams) -> None:
        trigger_position = self._probe.perform_probe()
        logger.info("Result is z=%.6f", trigger_position)
        self.last_trigger_position = trigger_position
