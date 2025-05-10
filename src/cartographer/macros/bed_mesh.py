from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, final

import numpy as np
from typing_extensions import override

from cartographer.interfaces.printer import Macro, MacroParams, Position, Sample, Toolhead
from cartographer.lib.nearest_neighbor import NearestNeighborSearcher

if TYPE_CHECKING:
    from cartographer.interfaces.multiprocessing import TaskExecutor
    from cartographer.probe import Probe
    from cartographer.probe.scan_mode import Model, ScanMode
    from cartographer.probe.touch_mode import TouchMode

logger = logging.getLogger(__name__)


class Configuration(Protocol):
    scan_speed: float
    scan_height: float
    scan_mesh_runs: int


@dataclass
class MeshPoint:
    x: float
    y: float
    include: bool


class MeshHelper(Protocol):
    def orig_macro(self, params: MacroParams) -> None: ...
    def prepare_scan_path(self, params: MacroParams) -> list[MeshPoint]: ...
    def prepare_touch_points(
        self,
        params: MacroParams,
        *,
        mesh_min: tuple[float, float] | None,
        mesh_max: tuple[float, float] | None,
    ) -> list[MeshPoint]: ...
    def finalize(self, offset: Position, positions: list[Position]): ...


@final
class BedMeshCalibrateMacro(Macro):
    name = "BED_MESH_CALIBRATE"
    description = "Gather samples across the bed to calibrate the bed mesh."

    def __init__(
        self,
        probe: Probe,
        toolhead: Toolhead,
        helper: MeshHelper,
        task_executor: TaskExecutor,
        config: Configuration,
    ) -> None:
        self.helper = helper
        self.probe = probe
        self.scan_mesh = _ScanMeshRunner(probe.scan, toolhead, task_executor, config)
        self.touch_mesh = _TouchMeshRunner(probe.touch, toolhead, config)

    @override
    def run(self, params: MacroParams) -> None:
        method = params.get("METHOD", default="scan").lower()
        if method != "scan" and method != "rapid_scan" and method != "touch":
            return self.helper.orig_macro(params)

        start_time = time.time()
        if method == "touch":
            boundaries = self.probe.touch.boundaries
            offset, positions = self.touch_mesh.run(
                params,
                self.helper.prepare_touch_points(
                    params,
                    mesh_min=(boundaries.min_x, boundaries.min_y),
                    mesh_max=(boundaries.max_x, boundaries.max_y),
                ),
            )
        else:
            offset, positions = self.scan_mesh.run(params, self.helper.prepare_scan_path(params))
        logger.debug("Bed mesh completed in %.2f seconds", time.time() - start_time)

        self.helper.finalize(offset, positions)


@final
class _TouchMeshRunner:
    def __init__(
        self,
        probe: TouchMode,
        toolhead: Toolhead,
        config: Configuration,
    ) -> None:
        self.probe = probe
        self.toolhead = toolhead
        self.config = config

    def run(self, params: MacroParams, points: list[MeshPoint]) -> tuple[Position, list[Position]]:
        speed = params.get_float("SPEED", default=self.config.scan_speed, minval=1)
        move_height = params.get_float("HORIZONTAL_MOVE_Z", default=self.config.scan_height, minval=1)
        if self.probe.model is None:
            msg = "cannot run bed mesh without a model"
            raise RuntimeError(msg)
        points = [p for p in points if p.include]

        for p in points:
            if not self.probe.is_within_boundaries(x=p.x, y=p.y):
                msg = f"probe point ({p.x:.2f},{p.y:.2f}) is outside of the touch boundaries "
                raise RuntimeError(msg)

        self.toolhead.move(z=move_height, speed=5)
        positions: list[Position] = []
        for p in points:
            self.toolhead.move(x=p.x, y=p.y, speed=speed)
            trigger_pos = self.probe.perform_probe()
            positions.append(Position(p.x, p.y, trigger_pos))

        return self.probe.offset, positions


@final
class _ScanMeshRunner:
    def __init__(
        self,
        probe: ScanMode,
        toolhead: Toolhead,
        task_executor: TaskExecutor,
        config: Configuration,
    ) -> None:
        self.probe = probe
        self.toolhead = toolhead
        self.task_executor = task_executor
        self.config = config

    def run(self, params: MacroParams, path: list[MeshPoint]) -> tuple[Position, list[Position]]:
        runs = params.get_int("RUNS", default=self.config.scan_mesh_runs, minval=1)
        speed = params.get_float("SPEED", default=self.config.scan_speed, minval=1)
        scan_height = params.get_float("HORIZONTAL_MOVE_Z", default=self.config.scan_height, minval=1)
        if self.probe.model is None:
            msg = "cannot run bed mesh without a model"
            raise RuntimeError(msg)

        self.toolhead.move(z=scan_height, speed=5)
        self._move_to_point(path[0], speed)

        start_time = time.time()
        with self.probe.start_session() as session:
            session.wait_for(lambda samples: len(samples) >= 5)
            for i in range(runs):
                is_odd = i & 1  # Bitwise check for odd numbers
                # Every other run should be going in reverse
                path_iter = reversed(path) if is_odd else path
                for point in path_iter:
                    self._move_to_point(point, speed)
                self.toolhead.dwell(0.250)
                self.toolhead.wait_moves()
            move_time = self.toolhead.get_last_move_time()
            session.wait_for(lambda samples: samples[-1].time >= move_time)
            count = len(session.items)
            session.wait_for(lambda samples: len(samples) >= count + 50)
        logger.debug("Bed scan completed in %.2f seconds", time.time() - start_time)

        samples = session.get_items()
        logger.debug("Gathered %d samples", len(samples))

        positions = self.task_executor.run(self._calculate_positions, self.probe.model, path, samples, scan_height)
        return self.probe.offset, positions

    def _move_to_point(self, point: MeshPoint, speed: float) -> None:
        offset = self.probe.offset
        self.toolhead.move(x=point.x - offset.x, y=point.y - offset.y, speed=speed)

    def _key(self, point: MeshPoint) -> tuple[float, float]:
        return round(point.x, 2), round(point.y, 2)

    def _calculate_positions(
        self, model: Model, path: list[MeshPoint], samples: list[Sample], scan_height: float
    ) -> list[Position]:
        included_points = [p for p in path if p.include]

        start_time = time.time()
        clusters = self._build_clusters(
            samples,
            included_points,
        )
        logger.debug("Sample clustering completed in %.2f seconds", time.time() - start_time)

        start_time = time.time()
        positions = [
            self._compute_position(
                (x, y),
                cluster,
                model,
                scan_height,
            )
            for (x, y), cluster in clusters.items()
        ]
        logger.debug("Cluster position computation completed in %.2f seconds", time.time() - start_time)
        return positions

    def _build_clusters(
        self,
        samples: list[Sample],
        points: list[MeshPoint],
    ) -> dict[tuple[float, float], list[Sample]]:
        offset = self.probe.offset
        searcher = NearestNeighborSearcher(points)
        cluster_map: dict[tuple[float, float], list[Sample]] = {self._key(p): [] for p in points}

        valid_samples: list[Sample] = []
        adjusted_positions: list[tuple[float, float]] = []

        for s in samples:
            if s.position is None:
                continue
            valid_samples.append(s)
            adjusted_positions.append(
                (
                    s.position.x + offset.x,
                    s.position.y + offset.y,
                )
            )

        nearest_points = searcher.batch_query(adjusted_positions)

        for s, point in zip(valid_samples, nearest_points):
            if point is not None and point.include:
                key = self._key(point)
                cluster_map[key].append(s)

        return cluster_map

    def _compute_position(
        self,
        key: tuple[float, float],
        cluster: list[Sample],
        model: Model,
        scan_height: float,
    ) -> Position:
        offset = self.probe.offset
        x, y = key
        if not cluster:
            msg = f"cluster ({x:.2f},{y:.2f}) has no samples"
            raise RuntimeError(msg)

        distances = list(map(lambda s: model.frequency_to_distance(s.frequency), cluster))
        median_distance = float(np.median(distances))

        if not math.isfinite(median_distance):
            msg = f"cluster ({x:.2f},{y:.2f}) has no valid samples"
            raise RuntimeError(msg)

        trigger_z = scan_height + self.probe.probe_height - median_distance
        pos = Position(x - offset.x, y - offset.y, trigger_z)
        return self.toolhead.apply_axis_twist_compensation(pos)
