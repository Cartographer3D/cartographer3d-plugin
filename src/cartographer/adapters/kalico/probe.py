from __future__ import annotations

from typing import TYPE_CHECKING, final

from cartographer.adapters.klipper_like.utils import reraise_for_klipper

if TYPE_CHECKING:
    from gcode import GCodeCommand

    from cartographer.interfaces.configuration import GeneralConfig
    from cartographer.interfaces.printer import ProbeMode, Toolhead
    from cartographer.macros.probe import ProbeMacro, QueryProbeMacro


@final
class KalicoCartographerProbe:
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
        self.sample_count = 1
        self.samples_tolerance = 0.1
        self.samples_retries = 0

    def get_offsets(self) -> tuple[float, float, float]:
        return self.probe.offset.as_tuple()

    def get_status(self, eventtime: float):
        del eventtime
        return {
            "name": "cartographer",
            "last_query": 1 if self.query_probe_macro.last_triggered else 0,
            "last_z_result": self.probe_macro.last_trigger_position or 0,
        }

    def get_lift_speed(self, gcmd: GCodeCommand | None = None):
        if gcmd is None:
            return self.lift_speed
        return gcmd.get_float("LIFT_SPEED", self.lift_speed, above=0.0)

    @reraise_for_klipper
    def run_probe(self, gcmd: GCodeCommand, *args: object, **kwargs: object) -> list[float]:
        del gcmd, args, kwargs
        pos = self.toolhead.get_position()
        trigger_pos = self.probe.perform_probe()
        return [pos.x, pos.y, trigger_pos]

    def multi_probe_begin(self):
        pass

    def multi_probe_end(self):
        pass
