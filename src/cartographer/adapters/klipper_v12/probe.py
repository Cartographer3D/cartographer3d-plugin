from __future__ import annotations

from typing import TYPE_CHECKING, final

from cartographer.adapters.klipper_like.utils import make_coord, reraise_for_klipper

if TYPE_CHECKING:
    from gcode import GCodeCommand

    from cartographer.interfaces.configuration import GeneralConfig
    from cartographer.interfaces.printer import ProbeMode, Toolhead
    from cartographer.macros.probe import ProbeMacro, QueryProbeMacro


@final
class KlipperV12CartographerProbe:
    def __init__(
        self,
        toolhead: Toolhead,
        probe: ProbeMode,
        probe_macro: ProbeMacro,
        query_probe_macro: QueryProbeMacro,
        config: GeneralConfig,
    ) -> None:
        self.probe = probe
        self.probe_macro = probe_macro
        self.query_probe_macro = query_probe_macro
        self.toolhead = toolhead
        self.lift_speed = config.lift_speed

    def multi_probe_begin(self) -> None:
        """Called before a sequence of probes. No-op for cartographer (contactless)."""

    def multi_probe_end(self) -> None:
        """Called after a sequence of probes. No-op for cartographer (contactless)."""

    def get_lift_speed(self, gcmd: GCodeCommand | None = None) -> float:
        """Return configured lift speed, optionally overridden by LIFT_SPEED gcmd param."""
        if gcmd is not None:
            return gcmd.get_float("LIFT_SPEED", self.lift_speed, above=0.0)
        return self.lift_speed

    def get_offsets(self) -> tuple[float, float, float]:
        """Return (x_offset, y_offset, z_offset) tuple."""
        return self.probe.offset.as_tuple()

    @reraise_for_klipper
    def run_probe(self, gcmd: GCodeCommand) -> list[float]:
        """Execute a single probe and return [x, y, z] position list."""
        del gcmd
        pos = self.toolhead.get_position()
        trigger_pos = self.probe.perform_probe()
        return [pos.x, pos.y, trigger_pos]

    def get_status(self, eventtime: float) -> dict[str, object]:
        del eventtime
        return {
            "name": "cartographer",
            "last_query": 1 if self.query_probe_macro.last_triggered else 0,
            "last_z_result": round(self.probe_macro.last_trigger_position or 0, 6),
            "last_probe_position": make_coord(self.probe_macro.last_probe_position),
        }
