from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from extras.probe import PrinterProbe
from gcode import CommandError
from typing_extensions import override

from cartographer.adapters.shared.utils import reraise_as

if TYPE_CHECKING:
    from gcode import GCodeCommand

    from cartographer.adapters.shared.printer import KlipperToolhead
    from cartographer.interfaces.printer import ProbeMode
    from cartographer.macros.probe import ProbeMacro, QueryProbeMacro


class ProbeStatus(TypedDict):
    name: str
    last_query: int
    last_z_result: float


class ProbeParams(TypedDict):
    probe_speed: float
    lift_speed: float


# TODO: Get the values from some configuration?
DEFAULT_LIFT_SPEED = 10
DEFAULT_PROBE_SPEED = 3


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
        pass


class KlipperCartographerProbe(PrinterProbe):
    def __init__(
        self,
        toolhead: KlipperToolhead,
        probe: ProbeMode,
        probe_macro: ProbeMacro,
        query_probe_macro: QueryProbeMacro,
    ) -> None:
        self.probe: ProbeMode = probe
        self.probe_macro: ProbeMacro = probe_macro
        self.query_probe_macro: QueryProbeMacro = query_probe_macro
        self.toolhead: KlipperToolhead = toolhead
        self.lift_speed: float = DEFAULT_LIFT_SPEED

    @override
    def get_probe_params(self, gcmd: GCodeCommand | None = None) -> ProbeParams:
        if gcmd is None:
            return ProbeParams(lift_speed=self.lift_speed, probe_speed=DEFAULT_PROBE_SPEED)

        lift_speed = gcmd.get_float("LIFT_SPEED", default=self.lift_speed, above=0)
        probe_speed = gcmd.get_float("SPEED", default=DEFAULT_PROBE_SPEED, above=0)
        return ProbeParams(lift_speed=lift_speed, probe_speed=probe_speed)

    @override
    def get_offsets(self) -> tuple[float, float, float]:
        return self.probe.offset.as_tuple()

    @override
    def get_status(self, eventtime: float) -> ProbeStatus:
        return ProbeStatus(
            name="cartographer",
            last_query=1 if self.query_probe_macro.last_triggered else 0,
            last_z_result=self.probe_macro.last_trigger_position or 0,
        )

    @override
    def start_probe_session(self, gcmd: GCodeCommand) -> KlipperProbeSession:
        return KlipperProbeSession(self.probe, self.toolhead)
