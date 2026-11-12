from __future__ import annotations

from typing import TYPE_CHECKING, final

from cartographer.adapters.klipper_like.utils import make_coord, reraise_for_klipper
from cartographer.interfaces.printer import Position

if TYPE_CHECKING:
    from gcode import GCodeCommand
    from klippy import Printer

    from cartographer.adapters.kalico.axis_twist_compensation import KalicoMeshAxisTwistCompensationAdapter
    from cartographer.adapters.klipper.endstop import KlipperEndstop
    from cartographer.interfaces.configuration import GeneralConfig
    from cartographer.interfaces.printer import ProbeMode, Toolhead
    from cartographer.macros.probe import ProbeMacro, QueryProbeMacro
    from cartographer.probe.touch_mode import TouchMode


@final
class KalicoCartographerProbe:
    def __init__(
        self,
        toolhead: Toolhead,
        probe: ProbeMode,
        probe_macro: ProbeMacro,
        query_probe_macro: QueryProbeMacro,
        config: GeneralConfig,
        *,
        printer: Printer | None = None,
        mcu_probe: KlipperEndstop | None = None,
    ) -> None:
        self.name = "cartographer"
        self.probe_name = "cartographer"
        self.is_default_probe = config.register_as_probe
        self.printer = printer
        if mcu_probe is not None:
            self.mcu_probe = mcu_probe

        self.probe = probe
        self.probe_macro = probe_macro
        self.query_probe_macro = query_probe_macro
        self.toolhead = toolhead

        self.lift_speed = config.lift_speed
        self.min_edge_distance = config.min_edge_distance
        self.sample_count = 1
        self.samples_tolerance = 0.1
        self.samples_retries = 0

    def get_offsets(self) -> tuple[float, float, float]:
        return self.probe.offset.as_tuple()

    def get_min_edge_distance(self) -> float | None:
        return self.min_edge_distance

    def get_status(self, eventtime: float):
        del eventtime
        return {
            "name": "cartographer",
            "last_query": 1 if self.query_probe_macro.last_triggered else 0,
            "last_z_result": round(self.probe_macro.last_trigger_position or 0, 6),
            "last_probe_position": make_coord(self.probe_macro.last_probe_position),
        }

    def get_lift_speed(self, gcmd: GCodeCommand | None = None):
        if gcmd is None:
            return self.lift_speed
        return gcmd.get_float("LIFT_SPEED", self.lift_speed, above=0.0)

    @reraise_for_klipper
    def run_probe(self, gcmd: GCodeCommand, *args: object, **kwargs: object) -> list[float]:
        del args, kwargs
        pos = self.toolhead.get_position()
        trigger_pos = self.probe.perform_probe(gcmd=gcmd)
        return [pos.x, pos.y, trigger_pos]

    def multi_probe_begin(self):
        pass

    def multi_probe_end(self):
        pass


@final
class KalicoTouchProbe:
    def __init__(
        self,
        toolhead: Toolhead,
        touch: TouchMode,
        config: GeneralConfig,
        compensation: KalicoMeshAxisTwistCompensationAdapter,
    ) -> None:
        self.name = self.probe_name = "cartographer_touch"
        self.is_default_probe = False
        self.toolhead = toolhead
        self.touch = touch
        self.compensation = compensation
        self.lift_speed = config.lift_speed
        self.sample_count = 1
        self.samples_tolerance = 0.1
        self.samples_retries = 0
        self.last_probe_position: Position | None = None

    def get_offsets(self) -> tuple[float, float, float]:
        return self.touch.offset.as_tuple()

    def get_status(self, eventtime: float) -> dict[str, object]:
        del eventtime
        position = self.last_probe_position
        return {
            "name": self.name,
            "last_z_result": round(position.z, 6) if position is not None else None,
            "last_probe_position": make_coord(position),
        }

    def get_lift_speed(self, gcmd: GCodeCommand | None = None) -> float:
        if gcmd is None:
            return self.lift_speed
        return gcmd.get_float("LIFT_SPEED", self.lift_speed, above=0.0)

    @reraise_for_klipper
    def run_probe(self, gcmd: GCodeCommand, *args: object, **kwargs: object) -> list[float]:
        del args, kwargs
        pos = self.toolhead.get_position()
        if not self.touch.boundaries.is_within(x=pos.x, y=pos.y):
            msg = f"Position ({pos.x:.2f}, {pos.y:.2f}) is outside touch boundaries"
            raise RuntimeError(msg)
        z = self.touch.perform_probe(gcmd=gcmd)
        z += self.compensation.get_z_compensation_value(x=pos.x, y=pos.y, probe_name=self.probe_name)
        self.last_probe_position = Position(pos.x, pos.y, z)
        return [pos.x, pos.y, z]

    def multi_probe_begin(self) -> None:
        pass

    def multi_probe_end(self) -> None:
        pass
