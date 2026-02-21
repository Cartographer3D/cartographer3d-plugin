from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from itertools import chain
from math import isfinite
from typing import TYPE_CHECKING, Literal, final

import numpy as np
from typing_extensions import override
from scipy.interpolate import griddata


from cartographer.interfaces.printer import (
    AxisTwistCompensation,
    Macro,
    MacroParams,
    Position,
    Sample,
    SupportsFallbackMacro,
    Toolhead,
)
from cartographer.lib.log import log_duration
from cartographer.macros.bed_mesh.helpers import (
    AdaptiveMeshCalculator,
    CoordinateTransformer,
    GridPointResult,
    MeshBounds,
    MeshGrid,
    Region,
    SampleProcessor,
)
from cartographer.macros.bed_mesh.paths.alternating_snake import AlternatingSnakePathGenerator
from cartographer.macros.bed_mesh.paths.random_path import RandomPathGenerator
from cartographer.macros.bed_mesh.paths.snake_path import SnakePathGenerator
from cartographer.macros.bed_mesh.paths.spiral_path import SpiralPathGenerator
from cartographer.macros.bed_mesh.paths.circular_snake_path import CircularSnakePathGenerator
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
    faulty_regions: list[Region]

    runs: int
    direction: Literal["x", "y"]
    height: float
    path: Literal["snake", "alternating_snake", "spiral", "random","circular_snake"]
    mesh_radius: float | None = None
    mesh_origin: tuple[float, float] | None = None
    round_probe_count: int | None = None

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
            path=config.scan.mesh_path,
            faulty_regions=list(map(lambda r: Region(r[0], r[1]), config.bed_mesh.faulty_regions)),
            mesh_radius=config.bed_mesh.mesh_radius,
            mesh_origin=config.bed_mesh.mesh_origin,
            round_probe_count=config.bed_mesh.round_probe_count,
        )


_directions: list[Literal["x", "y"]] = ["x", "y"]

PATH_GENERATOR_MAP = {
    "snake": SnakePathGenerator,
    "alternating_snake": AlternatingSnakePathGenerator,
    "spiral": SpiralPathGenerator,
    "random": RandomPathGenerator,
    "circular_snake": CircularSnakePathGenerator,
}


@dataclass
class MeshScanParams:
    mesh_bounds: MeshBounds
    resolution: tuple[int, int]
    speed: float
    height: float
    runs: int
    adaptive: bool
    adaptive_margin: float
    profile: str | None
    path_generator: PathGenerator
    mesh_radius: float | None = None
    mesh_origin: tuple[float, float] | None = None
    @classmethod
    def from_macro_params(
        cls, params: MacroParams, config: BedMeshCalibrateConfiguration, adapter: BedMeshAdapter
    ) -> MeshScanParams:
        """Create parameters from macro input and configuration."""

        # Initialize for both rectangular and circular cases
        radius = None
        origin = None

        # User can only override mesh_radius if it's defined in config.
        radius_val = config.mesh_radius
        if radius_val is not None and radius_val > 0.0:
            # Round bed mesh
            radius = params.get_float('MESH_RADIUS', default=config.mesh_radius, minval=0)

            # Validate correct radius to be defined.
            if radius is None or radius <= 0:
                radius = radius_val
                logger.warning("MESH_RADIUS parameter is invalid or not defined. Using config value of %.2f", radius)

            origin = get_float_tuple(
                params, 'MESH_ORIGIN', default=config.mesh_origin or (0.0, 0.0)
            )
            base_bounds = MeshBounds(
                (origin[0] - radius, origin[1] - radius),
                (origin[0] + radius, origin[1] + radius),
            )

            round_probe_count = params.get_int('ROUND_PROBE_COUNT', default=config.round_probe_count or 15)

            # Ensure odd number of probe points for round beds
            # This should be already validated in config parsing, but we enforce it here as well for safety
            round_probe_count = round_probe_count if round_probe_count % 2 == 1 else round_probe_count + 1

            base_resolution = (round_probe_count, round_probe_count)
        else:
            base_bounds = MeshBounds(
                get_float_tuple(params, "MESH_MIN", default=config.mesh_min),
                get_float_tuple(params, "MESH_MAX", default=config.mesh_max),
            )
            base_resolution = get_int_tuple(params, "PROBE_COUNT", default=config.probe_count)

        adaptive = params.get_int("ADAPTIVE", default=0) != 0
        adaptive_margin = params.get_float("ADAPTIVE_MARGIN", config.adaptive_margin, minval=0)

        # Calculate actual bounds and resolution
        #Todo: Modify adaptive mesh calculator to support round beds, using only full points for now.
        if adaptive and (radius is None or radius <= 0):
            calculator = AdaptiveMeshCalculator(base_bounds, base_resolution)
            object_points = list(chain.from_iterable(adapter.get_objects()))
            mesh_bounds = calculator.calculate_adaptive_bounds(object_points, adaptive_margin)
            resolution = calculator.calculate_adaptive_resolution(mesh_bounds)
            profile = None  # Adaptive meshes don't use profiles
        else:
            mesh_bounds = base_bounds
            resolution = base_resolution
            profile = params.get("PROFILE", default="default")

        # Create path generator
        direction: Literal["x", "y"] = get_choice(params, "DIRECTION", _directions, default=config.direction)
        if radius is not None and radius > 0:
           path_type = "circular_snake"
        else:
            path_type = get_choice(params, "PATH", default=config.path, choices=PATH_GENERATOR_MAP.keys())
        path_generator = PATH_GENERATOR_MAP[path_type](direction)

        return cls(
            mesh_bounds=mesh_bounds,
            resolution=resolution,
            speed=params.get_float("SPEED", default=config.speed, minval=50),
            height=params.get_float("HEIGHT", default=config.height, minval=0.5, maxval=5),
            runs=params.get_int("RUNS", default=config.runs, minval=1),
            adaptive=adaptive,
            adaptive_margin=adaptive_margin,
            profile=profile,
            path_generator=path_generator,
            mesh_radius = radius,
            mesh_origin = origin,
        )


@final
class BedMeshCalibrateMacro(Macro, SupportsFallbackMacro):
    description = "Gather samples across the bed to calibrate the bed mesh."

    def __init__(
        self,
        probe: Probe,
        toolhead: Toolhead,
        adapter: BedMeshAdapter,
        axis_twist_compensation: AxisTwistCompensation | None,
        task_executor: TaskExecutor,
        config: BedMeshCalibrateConfiguration,
    ):
        self.probe = probe
        self.toolhead = toolhead
        self.adapter = adapter
        self.task_executor = task_executor
        self.config = config
        self.coordinate_transformer = CoordinateTransformer(probe.scan.offset)
        self.axis_twist_compensation = axis_twist_compensation
        self._fallback: Macro | None = None

    @override
    def set_fallback_macro(self, macro: Macro) -> None:
        self._fallback = macro

    @override
    def run(self, params: MacroParams) -> None:
        """Main entry point for bed mesh calibration."""
        # Handle fallback for non-scan methods
        method = params.get("METHOD", "scan")
        if method.lower() != "scan":
            if self._fallback is None:
                msg = f"Bed mesh calibration method '{method}' not supported"
                raise RuntimeError(msg)
            return self._fallback.run(params)

        # Parse parameters and validate
        scan_params = MeshScanParams.from_macro_params(params, self.config, self.adapter)

        # Create mesh grid and processors
        grid = MeshGrid(
            scan_params.mesh_bounds.min_point,
            scan_params.mesh_bounds.max_point,
            scan_params.resolution[0],
            scan_params.resolution[1],
            scan_params.mesh_radius,
            scan_params.mesh_origin,
        )
        # Generate path and collect samples
        path = self._generate_path(grid, scan_params)
        self.adapter.clear_mesh()
        samples = self._collect_samples(path, scan_params)

        # Process samples and create mesh
        positions = self.task_executor.run(self._process_samples_to_positions, grid, samples, scan_params.height)

        # Fill gaps in circular mesh if needed
        if scan_params.mesh_radius is not None and scan_params.mesh_radius > 0:
            self._log_positions_debug("Circular mesh positions before gap fill", positions)
            positions = self._fill_circular_mesh_gaps(positions, grid) #As this now converts positions to rectangular mesh we can get
                                                                        #rid of changes in apply_zero_reference_height and apply_mesh
            self._log_positions_debug("Circular mesh positions after gap fill", positions)
        positions = self._apply_zero_reference_height(positions, scan_params, grid)

        # Apply mesh to adapter
        self.adapter.apply_mesh(positions, scan_params.profile)

    def _apply_zero_reference_height(
        self, positions: list[Position], params: MeshScanParams, grid: MeshGrid
    ) -> list[Position]:
        zrp = self.config.zero_reference_position
        if grid.contains_point(zrp):
            return self.coordinate_transformer.normalize_to_zero_reference_point(positions, zero_ref=zrp)

        self._move_probe_to_point(zrp, params.speed)
        zero_measure = params.height - self.probe.scan.measure_distance()
        nx, ny = self.coordinate_transformer.probe_to_nozzle(zrp)
        if self.axis_twist_compensation:
            zero_measure += self.axis_twist_compensation.get_z_compensation_value(x=float(nx), y=float(ny))

        return self.coordinate_transformer.normalize_to_zero_reference_point(positions, zero_height=zero_measure)

    def _generate_path(self, grid: MeshGrid, params: MeshScanParams) -> list[Point]:
        """Generate scanning path from grid points."""
        mesh_points = grid.generate_points()

        x_min, x_max = self.toolhead.get_axis_limits("x")
        y_min, y_max = self.toolhead.get_axis_limits("y")
        ox, oy = self.probe.scan.offset.x, self.probe.scan.offset.y

        return list(
            params.path_generator.generate_path(
                mesh_points,
                (x_min + max(0, ox), x_max + min(0, ox)),
                (y_min + max(0, oy), y_max + min(0, oy)),
            )
        )

    @log_duration("Collecting samples along the scanning path")
    def _collect_samples(self, path: list[Point], params: MeshScanParams) -> list[Sample]:
        """Collect samples by following the scanning path."""
        # Move to starting position
        self.toolhead.move(z=params.height, speed=5)
        self._move_probe_to_point(path[0], params.speed)
        self.toolhead.wait_moves()

        # Execute scan
        with self.probe.scan.start_session() as session:
            session.wait_for(lambda samples: len(samples) >= 10)

            for run_index in range(params.runs):
                sequence = path if run_index % 2 == 0 else reversed(path)
                for point in sequence:
                    self._move_probe_to_point(point, params.speed)

                self.toolhead.dwell(0.250)
                self.toolhead.wait_moves()

            # Wait for final samples
            move_time = self.toolhead.get_last_move_time()
            session.wait_for(lambda samples: samples[-1].time >= move_time)
            count = len(session.items)
            session.wait_for(lambda samples: len(samples) >= count + 10)

        samples = session.get_items()
        logger.debug("Collected %d samples across %d runs", len(samples), params.runs)
        return [self._transform_sample(s) for s in samples]

    def _move_probe_to_point(self, point: Point, speed: float) -> None:
        """Move probe to specified point (converts to nozzle coordinates)."""
        x, y = self.coordinate_transformer.probe_to_nozzle(point)
        self.toolhead.move(x=float(x), y=float(y), speed=speed)

    def _transform_sample(self, sample: Sample) -> Sample:
        """Transform sample to probe coordinates."""
        if sample.position is None:
            return sample

        probe_position = self.coordinate_transformer.nozzle_to_probe((sample.position.x, sample.position.y))
        return replace(
            sample, position=Position(x=float(probe_position[0]), y=float(probe_position[1]), z=sample.position.z)
        )

    @log_duration("Processing samples into final mesh positions")
    def _process_samples_to_positions(self, grid: MeshGrid, samples: list[Sample], height: float) -> list[Position]:
        """Process samples into final mesh positions."""
        sample_processor = SampleProcessor(grid)

        # Assign samples to grid points
        results = sample_processor.assign_samples_to_grid(samples, self.probe.scan.calculate_sample_distance)

        # Convert results to positions
        positions = self._results_to_positions(results, height)
        return self.coordinate_transformer.apply_faulty_regions(positions, self.config.faulty_regions)

    def _results_to_positions(self, results: list[GridPointResult], height: float) -> list[Position]:
        """Convert grid results to Position objects."""
        positions: list[Position] = []

        for result in results:
            rx, ry = result.point
            if not isfinite(result.z):
                msg = f"Grid point ({rx:.2f},{ry:.2f}) has no valid samples"
                raise RuntimeError(msg)

            # Calculate compensated height
            z = height - result.z
            nx, ny = self.coordinate_transformer.probe_to_nozzle(result.point)
            if self.axis_twist_compensation:
                z += self.axis_twist_compensation.get_z_compensation_value(x=float(nx), y=float(ny))

            # Convert back to probe coordinates
            positions.append(Position(x=float(rx), y=float(ry), z=z))

        return positions

    def _log_positions_debug(self, message: str, positions: list[Position]) -> None:
        if not logger.isEnabledFor(logging.DEBUG):
            return

        formatted_positions = [
            (
                round(position.x, 4),
                round(position.y, 4),
                "nan" if np.isnan(position.z) else round(position.z, 6),
            )
            for position in positions
        ]
        logger.debug("%s (%d points): %s", message, len(positions), formatted_positions)

    def _fill_circular_mesh_gaps(self, positions: list[Position], grid: MeshGrid) -> list[Position]:
        """Fill NaN gaps in circular mesh by replicating row edge values.
        
        This converts a circular mesh (with NaN values outside the circle) to a complete
        rectangular mesh by repeating leftmost/rightmost values in each row (matches Klipper behavior).
        """
        if not positions:
            return positions

        # Check if there are any NaN values to fill
        has_nan = any(np.isnan(p.z) for p in positions)
        if not has_nan:
            # No gaps to fill
            return positions

        # Convert positions to a matrix
        arr = np.asarray([(p.x, p.y, p.z) for p in positions])
        xs = np.unique(arr[:, 0])
        ys = np.unique(arr[:, 1])
        
        # create full grid with NaNs
        matrix = np.full((len(ys), len(xs)), np.nan)
        
        # Create lookup maps for efficient coordinate → index conversion
        x_index_map = {float(x): i for i, x in enumerate(xs)}
        y_index_map = {float(y): i for i, y in enumerate(ys)}
        
        # Populate z values at their corresponding grid positions
        for position in positions:
            x_grid_idx = x_index_map.get(position.x)
            y_grid_idx = y_index_map.get(position.y)
            
            if x_grid_idx is not None and y_grid_idx is not None:
                matrix[y_grid_idx, x_grid_idx] = position.z
        
        # Fill NaN values if any exist (row-by-row like Klipper)
        if np.isnan(matrix).any():
            # Process each row independently
            for j in range(len(ys)):
                row = matrix[j, :]
                
                # Find first and last valid (non-NaN) indices in this row
                valid_indices = np.where(~np.isnan(row))[0]
                
                if len(valid_indices) == 0:
                    # Entire row is NaN - shouldn't happen in circular mesh
                    continue
                
                first_valid_idx = valid_indices[0]
                last_valid_idx = valid_indices[-1]
                
                # Fill left side gaps by repeating leftmost valid value
                if first_valid_idx > 0:
                    row[0:first_valid_idx] = row[first_valid_idx]
                
                # Fill right side gaps by repeating rightmost valid value
                if last_valid_idx < len(row) - 1:
                    row[last_valid_idx + 1:] = row[last_valid_idx]

        # Safety check
        if np.isnan(matrix).any():
            msg = "Mesh has missing points that could not be extrapolated"
            raise RuntimeError(msg)
        
        # Convert back to list of positions (in same order as grid: y-major, x-minor)
        filled_positions: list[Position] = [
            Position(x=float(x), y=float(y), z=float(matrix[j, i]))
            for j, y in enumerate(ys)
            for i, x in enumerate(xs)
        ]
        
        # Count how many NaN values were filled
        nan_count_before = sum(1 for p in positions if np.isnan(p.z))
        logger.info("Filled %d circular mesh gaps by replicating row edge values",  nan_count_before)
        
        return filled_positions
