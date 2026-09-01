from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, final

from typing_extensions import override

from cartographer.interfaces.configuration import TouchConfig
from cartographer.interfaces.printer import Macro, MacroParams, Position, Toolhead
from cartographer.macros.fields import config_ref, param, parse

if TYPE_CHECKING:
    from cartographer.probe.touch_mode import TouchMode


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TouchProbeMacroParams:
    """Parameters for CARTOGRAPHER_TOUCH_PROBE."""

    max_samples: int = param(
        "Per-call maximum number of touch attempts before giving up."
        " Overrides the configured max_samples for this probe call only.",
        default=config_ref(TouchConfig, "max_samples"),
        min=1,
    )


@final
class TouchProbeMacro(Macro):
    description = "Touch the bed to get the height offset at the current position."
    last_trigger_position: float | None = None
    last_probe_position: Position | None = None

    def __init__(self, probe: TouchMode, toolhead: Toolhead, *, max_samples: int) -> None:
        self._probe = probe
        self._toolhead = toolhead
        self._max_samples = max_samples

    @override
    def run(self, params: MacroParams) -> None:
        p = parse(TouchProbeMacroParams, params, max_samples=self._max_samples)
        trigger_pos = self._probe.perform_probe(max_samples=p.max_samples)
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
