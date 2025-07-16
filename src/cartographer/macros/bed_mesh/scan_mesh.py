from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cached_property
from itertools import chain
from math import ceil, isfinite
from typing import TYPE_CHECKING, Literal, final

import numpy as np
from typing_extensions import override

from cartographer.interfaces.printer import Macro, MacroParams, Position, Sample, SupportsFallbackMacro, Toolhead
from cartographer.lib.log import log_duration
from cartographer.macros.bed_mesh.alternating_snake import AlternatingSnakePathGenerator
from cartographer.macros.bed_mesh.mesh_utils import assign_samples_to_grid, normalize_to_zero_reference
from cartographer.macros.bed_mesh.random_path import RandomPathGenerator
from cartographer.macros.bed_mesh.snake_path import SnakePathGenerator
from cartographer.macros.bed_mesh.spiral_path import SpiralPathGenerator
from cartographer.macros.utils import get_choice, get_float_tuple, get_int_tuple

if TYPE_CHECKING:
    from cartographer.interfaces.configuration import Configuration
    from cartographer.interfaces.multiprocessing import TaskExecutor
    from cartographer.macros.bed_mesh.interfaces import BedMeshAdapter, PathGenerator, Point
    from cartographer.probe import Probe

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BedMeshCalibrateConfiguration:
    mesh_min: tuple[float, float]
    mesh_max: tuple[float, float]
    probe_count: tuple[int, int]
    speed: float
    adaptive_margin: float
    zero_reference_position: Point

    runs: int
    direction: Literal["x", "y"]
    height: float
    corner_radius: float
    path: Literal["snake", "alternating_snake", "spiral", "random"]

    @staticmethod
    def from_config(config: Configuration):
        return BedMeshCalibrateConfiguration(
            mesh_min=config.bed_mesh.mesh_min,
            mesh_max=config.bed_mesh.mesh_max,
            probe_count=config.bed_mesh.probe_count,
            speed=config.bed_mesh.speed,
            adaptive_margin=config.bed_mesh.adaptive_margin,
            zero_reference_position=config.bed_mesh.zero_reference_position,
            runs=config.scan.mesh_runs,
            direction=config.scan.mesh_direction,
            height=config.scan.mesh_height,
            corner_radius=config.scan.mesh_corner_radius,
            path=config.scan.mesh_path,
        )


_directions: list[Literal["x", "y"]] = ["x", "y"]

PATH_GENERATOR_MAP = {
    "snake": SnakePathGenerator,
    "alternating_snake": AlternatingSnakePathGenerator,
    "spiral": SpiralPathGenerator,
    "random": RandomPathGenerator,
}


@dataclass
class BedMeshParams:
    _mesh_min: tuple[float, float]
    _mesh_max: tuple[float, float]
    adaptive_margin: float
    speed: float
    runs: int
    height: float
    corner_radius: float
    direction: Literal["x", "y"]
    path_generator: PathGenerator
    adaptive: bool
    _probe_count: tuple[int, int]
    profile: str | None

    adapter: BedMeshAdapter

    @staticmethod
    def from_macro_params(
        params: MacroParams, config: BedMeshCalibrateConfiguration, adapter: BedMeshAdapter
    ) -> BedMeshParams:
        direction: Literal["x", "y"] = get_choice(params, "DIRECTION", _directions, default=config.direction)
        corner_radius = params.get_float("CORNER_RADIUS", default=config.corner_radius, minval=0)
        path_type = get_choice(params, "PATH", default=config.path, choices=PATH_GENERATOR_MAP.keys())
        path_generator = PATH_GENERATOR_MAP[path_type](direction, corner_radius)
        adaptive = params.get_int("ADAPTIVE", default=0) != 0

        return BedMeshParams(
            _mesh_min=get_float_tuple(params, "MESH_MIN", default=config.mesh_min),
            _mesh_max=get_float_tuple(params, "MESH_MAX", default=config.mesh_max),
            adaptive_margin=params.get_float("ADAPTIVE_MARGIN", config.adaptive_margin, minval=0),
            speed=params.get_float("SPEED", default=config.speed, minval=50),
            runs=params.get_int("RUNS", default=config.runs, minval=1),
            height=params.get_float("HEIGHT", default=config.height, minval=0.5, maxval=5),
            corner_radius=corner_radius,
            direction=direction,
            path_generator=path_generator,
            adaptive=adaptive,
            _probe_count=get_int_tuple(params, "PROBE_COUNT", default=config.probe_count),
            profile=params.get("PROFILE", default="default" if not adaptive else None),
            adapter=adapter,
        )

    @cached_property
    def adaptive_bounds(self) -> tuple[Point, Point]:
        if not self.adaptive:
            return self._mesh_min, self._mesh_max

        points = list(chain.from_iterable(self.adapter.get_objects()))
        if not points:
            return self._mesh_min, self._mesh_max

        margin = self.adaptive_margin

        min_x = min(x for (x, _) in points)
        max_x = max(x for (x, _) in points)
        min_y = min(y for (_, y) in points)
        max_y = max(y for (_, y) in points)

        obj_min_x = max(min_x - margin, self._mesh_min[0])
        obj_max_x = min(max_x + margin, self._mesh_max[0])
        obj_min_y = max(min_y - margin, self._mesh_min[1])
        obj_max_y = min(max_y + margin, self._mesh_max[1])

        return (obj_min_x, obj_min_y), (obj_max_x, obj_max_y)

    @cached_property
    def adaptive_resolution(self) -> tuple[int, int]:
        orig_min = self._mesh_min
        orig_max = self._mesh_max
        orig_x_res, orig_y_res = self._probe_count
        adapted_min, adapted_max = self.adaptive_bounds

        orig_width = orig_max[0] - orig_min[0]
        orig_height = orig_max[1] - orig_min[1]
        x_density = (orig_x_res - 1) / orig_width if orig_width else 0
        y_density = (orig_y_res - 1) / orig_height if orig_height else 0

        adapted_width = adapted_max[0] - adapted_min[0]
        adapted_height = adapted_max[1] - adapted_min[1]

        x_res = max(MIN_POINTS, ceil(adapted_width * x_density) + 1)
        y_res = max(MIN_POINTS, ceil(adapted_height * y_density) + 1)

        return x_res, y_res

    def is_point_within_bounds(self, point: Point) -> bool:
        mesh_min, mesh_max = self.adaptive_bounds
        x, y = point
        return bool(mesh_min[0] <= x <= mesh_max[0] and mesh_min[1] <= y <= mesh_max[1])


MIN_POINTS = 3


@final
class BedMeshCalibrateMacro(Macro, SupportsFallbackMacro):
    description = "Gather samples across the bed to calibrate the bed mesh."

    _fallback: Macro | None = None

    def __init__(
        self,
        probe: Probe,
        toolhead: Toolhead,
        adapter: BedMeshAdapter,
        task_executor: TaskExecutor,
        config: BedMeshCalibrateConfiguration,
    ) -> None:
        self.probe = probe
        self.toolhead = toolhead
        self.adapter = adapter
        self.task_executor = task_executor
        self.config = config

    @override
    def set_fallback_macro(self, macro: Macro) -> None:
        self._fallback = macro

    @override
    def run(self, params: MacroParams) -> None:
        method = params.get("METHOD", "scan")
        if method.lower() != "scan":
            if self._fallback is None:
                msg = f"Bed mesh calibration method '{method}' not supported"
                raise RuntimeError(msg)
            return self._fallback.run(params)

        parsed_params = BedMeshParams.from_macro_params(params, self.config, self.adapter)
        if not parsed_params.is_point_within_bounds(self.config.zero_reference_position):
            x, y = self.config.zero_reference_position
            zrp = f"({x:.2f},{y:.2f})"
            (min_x, min_y), (max_x, max_y) = parsed_params.adaptive_bounds
            bounds = f"({min_x:.2f},{min_y:.2f}) - ({max_x:.2f},{max_y:.2f})"
            msg = f"Zero reference position {zrp} must be within the mesh bounds {bounds}"
            raise RuntimeError(msg)

        mesh_points = self._generate_mesh_points(parsed_params)
        path = list(parsed_params.path_generator.generate_path(mesh_points))

        self.adapter.clear_mesh()
        samples = self._sample_path(parsed_params, path)
        positions = self.task_executor.run(self._assign_positions_to_points, mesh_points, samples, parsed_params.height)

        self.adapter.apply_mesh(positions, parsed_params.profile)

    def _generate_mesh_points(
        self,
        params: BedMeshParams,
    ) -> list[Point]:
        adapted_min, adapted_max = params.adaptive_bounds
        x_res, y_res = params.adaptive_resolution

        x_points = np.round(np.linspace(adapted_min[0], adapted_max[0], x_res), 2)
        y_points = np.round(np.linspace(adapted_min[1], adapted_max[1], y_res), 2)

        mesh = [(x, y) for x in x_points for y in y_points]  # shape: [y][x]
        return mesh

    @log_duration("Bed scan")
    def _sample_path(self, params: BedMeshParams, path: list[Point]) -> list[Sample]:
        runs = params.runs
        height = params.height
        speed = params.speed

        self.toolhead.move(z=height, speed=5)
        self._move_probe_to_point(path[0], speed)
        self.toolhead.wait_moves()

        with self.probe.scan.start_session() as session:
            session.wait_for(lambda samples: len(samples) >= 10)
            for i in range(runs):
                sequence = path if i % 2 == 0 else reversed(path)
                for point in sequence:
                    self._move_probe_to_point(point, speed)
                self.toolhead.dwell(0.250)
                self.toolhead.wait_moves()
            move_time = self.toolhead.get_last_move_time()
            session.wait_for(lambda samples: samples[-1].time >= move_time)
            count = len(session.items)
            session.wait_for(lambda samples: len(samples) >= count + 10)

        samples = session.get_items()
        logger.debug("Gathered %d samples", len(samples))
        return samples

    def _probe_point_to_nozzle_point(self, point: Point) -> Point:
        x, y = point
        offset = self.probe.scan.offset
        return (x - offset.x, y - offset.y)

    def _nozzle_point_to_probe_point(self, point: Point) -> Point:
        x, y = point
        offset = self.probe.scan.offset
        return (x + offset.x, y + offset.y)

    def _move_probe_to_point(self, point: Point, speed: float) -> None:
        x, y = self._probe_point_to_nozzle_point(point)
        self.toolhead.move(x=float(x), y=float(y), speed=speed)

    @log_duration("Cluster position computation")
    def _assign_positions_to_points(
        self, mesh_points: list[Point], samples: list[Sample], height: float
    ) -> list[Position]:
        nozzle_points = [self._probe_point_to_nozzle_point(p) for p in mesh_points]
        results = assign_samples_to_grid(nozzle_points, samples, self.probe.scan.calculate_sample_distance)

        positions: list[Position] = []
        for result in results:
            rx, ry = result.point
            if not isfinite(result.z):
                msg = f"Cluster ({rx:.2f},{ry:.2f}) has no valid samples"
                raise RuntimeError(msg)

            z = height - result.z
            compensated = self.toolhead.apply_axis_twist_compensation(Position(x=float(rx), y=float(ry), z=z))
            px, py = self._nozzle_point_to_probe_point(result.point)
            positions.append(Position(x=float(px), y=float(py), z=compensated.z))

        return normalize_to_zero_reference(positions, self.config.zero_reference_position)
