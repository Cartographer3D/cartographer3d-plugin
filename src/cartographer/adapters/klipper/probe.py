from __future__ import annotations

from typing import TYPE_CHECKING, final

from extras import manual_probe

from cartographer.adapters.klipper_like.utils import reraise_for_klipper

if TYPE_CHECKING:
    from gcode import GCodeCommand

    from cartographer.interfaces.configuration import GeneralConfig
    from cartographer.interfaces.printer import ProbeMode, Toolhead
    from cartographer.macros.probe import ProbeMacro, QueryProbeMacro


class KlipperProbeSession:
    def __init__(self, probe: ProbeMode, toolhead: Toolhead) -> None:
        self._probe: ProbeMode = probe
        self._results: list[list[float]] = []
        self.toolhead: Toolhead = toolhead

    @reraise_for_klipper
    def run_probe(self, gcmd: GCodeCommand) -> None:
        del gcmd
        pos = self.toolhead.get_position()
        trigger_pos = self._probe.perform_probe()
        self._results.append([pos.x, pos.y, trigger_pos])

    def pull_probed_results(self):
        results = self._results
        self._results = []

        # Return ProbeResult objects if available (Klipper >= v0.13.0-465), otherwise return lists
        # for backward compatibility with older Klipper versions

        # Check if ProbeResult exists (introduced in Dec 2025)
        if not hasattr(manual_probe, "ProbeResult"):
            # Older Klipper versions expect plain lists
            return results

        offset = self._probe.offset
        return [
            manual_probe.ProbeResult(
                px + offset.x,
                py + offset.y,
                pz - offset.z,
                px,
                py,
                pz,
            )
            for [px, py, pz] in results
        ]

    def end_probe_session(self) -> None:
        self._results = []


@final
class KlipperCartographerProbe:
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

    def get_offsets(self, gcmd: GCodeCommand | None = None) -> tuple[float, float, float]:
        del gcmd
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
