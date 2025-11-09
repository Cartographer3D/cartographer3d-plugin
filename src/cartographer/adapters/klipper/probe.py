from __future__ import annotations

from typing import TYPE_CHECKING, final

from gcode import CommandError

from cartographer.adapters.utils import reraise_as

if TYPE_CHECKING:
    from gcode import GCodeCommand

    from cartographer.adapters.klipper.toolhead import KlipperToolhead
    from cartographer.interfaces.configuration import GeneralConfig
    from cartographer.interfaces.printer import ProbeMode
    from cartographer.macros.probe import ProbeMacro, QueryProbeMacro


class KlipperProbeSession:
    def __init__(self, probe: ProbeMode, toolhead: KlipperToolhead) -> None:
        self._probe: ProbeMode = probe
        self._results: list[list[float]] = []
        self.toolhead: KlipperToolhead = toolhead

    @reraise_as(CommandError)
    def run_probe(self, gcmd: GCodeCommand) -> None:
        del gcmd
        pos = self.toolhead.get_position()
        trigger_pos = self._probe.perform_probe()
        self._results.append([pos.x, pos.y, trigger_pos])

    def pull_probed_results(self):
        result = self._results
        self._results = []
        return result

    def end_probe_session(self) -> None:
        self._results = []


@final
class KlipperCartographerProbe:
    def __init__(
        self,
        toolhead: KlipperToolhead,
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

    def _get_lift_speed(self, gcmd: GCodeCommand | None = None):
        if gcmd is None:
            return self.lift_speed
        return gcmd.get_float("LIFT_SPEED", self.lift_speed, above=0.0)

    def get_probe_params(self, gcmd: GCodeCommand | None = None):
        return {
            "probe_speed": 5,
            "lift_speed": self._get_lift_speed(gcmd),
            "samples": 1,
            "sample_retract_dist": 0.2,
            "samples_tolerance": 0.1,
            "samples_tolerance_retries": 0,
            "samples_result": "median",
        }

    def get_offsets(self) -> tuple[float, float, float]:
        return self.probe.offset.as_tuple()

    def get_status(self, eventtime: float):
        del eventtime
        return {
            "name": "cartographer",
            "last_query": 1 if self.query_probe_macro.last_triggered else 0,
            "last_z_result": self.probe_macro.last_trigger_position or 0,
        }

    def start_probe_session(self, gcmd: GCodeCommand) -> KlipperProbeSession:
        del gcmd
        return KlipperProbeSession(self.probe, self.toolhead)
