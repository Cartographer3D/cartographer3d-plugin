from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from math import ceil
from typing import TYPE_CHECKING, Callable, Sequence, final, overload

import numpy as np
from typing_extensions import override

from cartographer.interfaces.printer import Position, Sample
from cartographer.lib import scipy_helpers
from cartographer.lib.scipy_helpers import rbf_interpolator

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from cartographer.macros.bed_mesh.interfaces import Point

MIN_GRID_RESOLUTION = 3
DEFAULT_MAX_SAMPLE_DISTANCE = 1.0

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Region:
    min_point: Point
    max_point: Point

    def contains_point(self, point: Point, epsilon: float = 1e-2) -> bool:
        """Check if a point is within the region bounds."""
        x, y = point
        return bool(
            self.min_point[0] - epsilon <= x <= self.max_point[0] + epsilon
            and self.min_point[1] - epsilon <= y <= self.max_point[1] + epsilon
        )


@dataclass(frozen=True)
class MeshGrid(Region):
    """Represents a 2D mesh grid with coordinate mappings and bounds."""

    min_point: Point
    max_point: Point
    x_resolution: int
    y_resolution: int
    mesh_radius: float | None = None
    mesh_origin: tuple[float, float] | None = None

    def __post_init__(self):
        if self.x_resolution < MIN_GRID_RESOLUTION or self.y_resolution < MIN_GRID_RESOLUTION:
            msg = f"Grid resolution must be at least {MIN_GRID_RESOLUTION}x{MIN_GRID_RESOLUTION}"
            raise ValueError(msg)

    @property
    def x_coords(self) -> NDArray[np.float_]:
        """Get array of x coordinates."""
        return np.round(np.linspace(self.min_point[0], self.max_point[0], self.x_resolution), 2)

    @property
    def y_coords(self) -> NDArray[np.float_]:
        """Get array of y coordinates."""
        return np.round(np.linspace(self.min_point[1], self.max_point[1], self.y_resolution), 2)

    @property
    def x_step(self) -> float:
        """Get step size in x direction."""
        return float(self.max_point[0] - self.min_point[0]) / (self.x_resolution - 1)

    @property
    def y_step(self) -> float:
        """Get step size in y direction."""
        return float(self.max_point[1] - self.min_point[1]) / (self.y_resolution - 1)

    def generate_points(self) -> list[Point]:
        """Generate all grid points in x-major order (y is inner loop).

        For circular meshes, only returns points within the circular region.
        """
        points: list[Point] = []
        for x in self.x_coords:
            for y in self.y_coords:
                pt = (float(x), float(y))
                if self.contains_point(pt):
                    points.append(pt)
        return points

    def point_to_grid_index(self, point: Point) -> tuple[int, int]:
        """Convert a point to grid indices (j, i) where j=row, i=col."""
        x, y = point
        i = round((x - self.min_point[0]) / self.x_step)
        j = round((y - self.min_point[1]) / self.y_step)
        return j, i

    def grid_index_to_point(self, j: int, i: int) -> Point:
        """Convert grid indices to a point."""
        x = self.x_coords[i]
        y = self.y_coords[j]
        return (float(x), float(y))

    def is_valid_index(self, j: int, i: int) -> bool:
        """Check if grid indices are valid."""
        return 0 <= i < self.x_resolution and 0 <= j < self.y_resolution

    @override
    def contains_point(self, point: Point, epsilon: float = 1e-2) -> bool:
        """Check whether a point lies within the grid area.

        For circular meshes: point is inside if distance to `mesh_origin` ≤ `mesh_radius` + epsilon.
        For rectangular meshes: point is inside if within min/max bounds + epsilon.
        """
        x, y = point

        # Circular mesh containment (takes precedence if defined)
        if self.mesh_radius is not None and self.mesh_radius >0 and self.mesh_origin is not None:
            ox, oy = self.mesh_origin
            dist_sq = (x - ox) ** 2 + (y - oy) ** 2
            radius_eps_sq = (self.mesh_radius + epsilon) ** 2
            return bool(dist_sq <= radius_eps_sq)

        # Rectangular containment (fallback)
        return bool(
            self.min_point[0] - epsilon <= x <= self.max_point[0] + epsilon
            and self.min_point[1] - epsilon <= y <= self.max_point[1] + epsilon
        )


@dataclass(frozen=True)
class GridPointResult:
    """Result of assigning samples to a grid point."""

    point: Point
    z: float
    sample_count: int


@final
class SampleProcessor:
    """Handles processing and assignment of samples to grid points."""

    def __init__(self, grid: MeshGrid, max_distance: float = DEFAULT_MAX_SAMPLE_DISTANCE):
        self.grid = grid
        self.max_distance = max_distance

    def assign_samples_to_grid(
        self,
        samples: list[Sample],
        calculate_height: Callable[[Sample], float],
    ) -> list[GridPointResult]:
        """Assign samples to grid points and calculate median heights."""
        accumulator: dict[tuple[int, int], list[float]] = defaultdict(list)
        sample_points = [
            (sample.position.x, sample.position.y, sample) for sample in samples if sample.position is not None
        ]

        for x, y, sample in sample_points:
            if not self.grid.contains_point((x, y)):
                continue

            j, i = self.grid.point_to_grid_index((x, y))
            if not self.grid.is_valid_index(j, i):
                continue

            grid_point = self.grid.grid_index_to_point(j, i)
            distance = np.hypot(x - grid_point[0], y - grid_point[1])
            if distance > self.max_distance:
                continue

            sample_height = calculate_height(sample)
            accumulator[(j, i)].append(sample_height)

        results: list[GridPointResult] = []
        
        # Process only grid indices that have samples or are valid mesh points
        for j in range(self.grid.y_resolution):
            for i in range(self.grid.x_resolution):
                grid_point = self.grid.grid_index_to_point(j, i)
                
                # Skipping points outside of circular mesh area if they don't have any samples.
                values = accumulator.get((j, i), [])
                if not values and not self.grid.contains_point(grid_point):
                    continue
                    
                count = len(values)

                z = float(np.median(values)) if values else np.nan
                results.append(GridPointResult(point=grid_point, z=z, sample_count=count))

        return results


@final
class CoordinateTransformer:
    """Handles coordinate transformations between different reference frames."""

    def __init__(self, probe_offset: Position):
        self.probe_offset = probe_offset

    def probe_to_nozzle(self, point: Point) -> Point:
        """Convert probe coordinates to nozzle coordinates."""
        x, y = point
        return (x - self.probe_offset.x, y - self.probe_offset.y)

    def nozzle_to_probe(self, point: Point) -> Point:
        """Convert nozzle coordinates to probe coordinates."""
        x, y = point
        return (x + self.probe_offset.x, y + self.probe_offset.y)

    @overload
    def normalize_to_zero_reference_point(self, positions: list[Position], *, zero_ref: Point) -> list[Position]: ...
    @overload
    def normalize_to_zero_reference_point(self, positions: list[Position], *, zero_height: float) -> list[Position]: ...
    def normalize_to_zero_reference_point(
        self,
        positions: list[Position],
        *,
        zero_ref: Point | None = None,
        zero_height: float | None = None,
    ) -> list[Position]:
        """Normalize positions to a zero reference point using bilinear interpolation
        for rectangular grids or scipy griddata for sparse circular meshes.
        """
        if not positions:
            return []
        # Convert to NumPy array (shape: (N, 3))
        arr: NDArray[np.float_] = np.asarray([(p.x, p.y, p.z) for p in positions])

        # Extract sorted unique coords
        xs = np.unique(arr[:, 0])
        ys = np.unique(arr[:, 1])

        # Check if rectangular grid is complete or if we have an sparse (circular) grid.
        if len(xs) * len(ys) == len(positions):
            # Sort positions in matching y-major order and reshape z
            # This works if positions form a complete rectangular grid
            # and can be sorted by (y, x)
            sort_idx = np.lexsort((arr[:, 0], arr[:, 1]))
            z_grid = arr[sort_idx, 2].reshape(len(ys), len(xs))
        else:
            #Irregular grid, don't attempt to reshape it to rectangular shape.
            z_grid = None  # Will handle interpolation differently for irregular grids

        # Determine zero reference height
        if zero_height is not None:
            z_ref = zero_height
        elif zero_ref is not None:
            if z_grid is not None:
                # Rectangular grid: use bilinear interpolation
                z_ref = self._bilinear_interpolate(xs, ys, z_grid, zero_ref)
            else:
                # Irregular grid (circular mesh): use scipy griddata interpolation
                if not scipy_helpers.is_available():
                    msg = "scipy is required for interpolation with irregular circular meshes"
                    raise RuntimeError(msg)
                from scipy.interpolate import griddata
                points = arr[:, :2]  # (x, y) coordinates
                values = arr[:, 2]   # z heights
                z_ref = float(griddata(points, values, [zero_ref], method='linear')[0])
        else:
            msg = "Either zero_ref or zero_height must be provided."
            raise ValueError(msg)

        # Normalize heights
        if z_grid is not None:
            # Rectangular grid: subtract from grid and rebuild
            z_grid -= z_ref
            normalized_positions = [
                Position(x=float(x), y=float(y), z=float(z)) for y, row in zip(ys, z_grid) for x, z in zip(xs, row)
            ]
        else:
            # Irregular grid: subtract from original positions directly
            normalized_positions = [
                Position(x=p.x, y=p.y, z=p.z - z_ref) for p in positions
            ]
        return normalized_positions

    def _bilinear_interpolate(
        self,
        xs: NDArray[np.float_],
        ys: NDArray[np.float_],
        z_grid: NDArray[np.float_],
        point: Point,
    ) -> float:
        """Perform bilinear interpolation to find height at a point."""
        dx = xs[1] - xs[0]
        dy = ys[1] - ys[0]

        tx = (point[0] - xs[0]) / dx
        ty = (point[1] - ys[0]) / dy

        # Clamp to valid indices within the grid
        max_x_idx = len(xs) - 1
        max_y_idx = len(ys) - 1

        xi = min(int(np.floor(tx)), max_x_idx - 1)  # Ensure xi+1 is valid
        yi = min(int(np.floor(ty)), max_y_idx - 1)  # Ensure yi+1 is valid

        tx -= xi
        ty -= yi

        z00 = z_grid[yi, xi]
        z01 = z_grid[yi, xi + 1]
        z10 = z_grid[yi + 1, xi]
        z11 = z_grid[yi + 1, xi + 1]

        z0 = (1 - tx) * z00 + tx * z01
        z1 = (1 - tx) * z10 + tx * z11
        z_ref = (1 - ty) * z0 + ty * z1

        return float(z_ref)

    def apply_faulty_regions(
        self,
        positions: list[Position],
        faulty_regions: list[Region],
    ) -> list[Position]:
        """Mask faulty regions in a rectangular heightmap and interpolate them using scipy if available."""
        if not positions:
            return []

        # Convert to array (N,3)
        arr: NDArray[np.float_] = np.asarray([(p.x, p.y, p.z) for p in positions])

        xs = np.unique(arr[:, 0])
        ys = np.unique(arr[:, 1])

        # For complete rectangular grids, sort and reshape directly
        # For sparse grids (circular), fill missing points with NaN first
        if len(xs) * len(ys) == len(positions):
            # Complete rectangular grid - use original approach
            sort_idx = np.lexsort((arr[:, 0], arr[:, 1]))
            z_grid = arr[sort_idx, 2].reshape(len(ys), len(xs))
        else:
            # Sparse grid (circular) - fill missing corners with NaN to make it complete
            z_grid = np.full((len(ys), len(xs)), np.nan)
            
            # Create lookup maps for efficient coordinate → index conversion
            x_index_map = {x: i for i, x in enumerate(xs)}
            y_index_map = {y: i for i, y in enumerate(ys)}
            
            # Populate z values at their corresponding grid positions
            for i in range(len(positions)):
                x_coord = arr[i, 0]
                y_coord = arr[i, 1]
                z_value = arr[i, 2]
                
                x_grid_idx = x_index_map.get(x_coord)
                y_grid_idx = y_index_map.get(y_coord)
                
                if x_grid_idx is not None and y_grid_idx is not None:
                    z_grid[y_grid_idx, x_grid_idx] = z_value

        # Build mask (ys major, xs minor)
        mask: NDArray[np.bool_] = np.zeros_like(z_grid, dtype=bool)
        for i in range(len(ys)):
            for j in range(len(xs)):
                point = (xs[j], ys[i])
                # Only mark as faulty if it's in a faulty region AND not already NaN (outside mesh radius)
                if not np.isnan(z_grid[i, j]) and any(region.contains_point(point) for region in faulty_regions):
                    mask[i, j] = True

        z_grid_masked = np.where(mask, np.nan, z_grid)

        # Interpolate missing values if scipy is available
        if np.any(mask):
            if not scipy_helpers.is_available():
                msg = "scipy is required for interpolation of faulty regions"
                raise RuntimeError(msg)
            logger.info("Interpolating %d faulty points", np.sum(mask))

            # Use the same order as flattening the mask/grid
            X, Y = np.meshgrid(xs, ys)  # noqa: N806
            points_grid = np.column_stack([X.ravel(), Y.ravel()])

            # For circular mesh we should also mask NaN values, this can happen if faulty region
            # overlaps with points outside of sparse mesh radius. 
            # This ensures we only interpolate based on valid points.
            nan_mask = np.isnan(z_grid_masked.ravel())
            combined_mask = mask.ravel() | nan_mask
            
            valid_points = points_grid[~combined_mask]
            valid_values = z_grid_masked.ravel()[~combined_mask]

            rbf = rbf_interpolator(valid_points, valid_values, neighbors=min(64, len(valid_points)), smoothing=0.0)

            missing_points = points_grid[mask.ravel()]
            interpolated = rbf(missing_points)  # pyright: ignore[reportUnknownVariableType]

            z_grid_masked[mask] = interpolated

        # We are now filtering out any remaining NaN values which is caused by rectangular zgrid
        # being filled with NaN for points outside of circular mesh radius.
        new_positions = [
            Position(x=float(x), y=float(y), z=float(z))
            for y, row in zip(ys, z_grid_masked)
            for x, z in zip(xs, row)
        ]
        return new_positions


@dataclass(frozen=True)
class MeshBounds:
    """Represents the bounds of a mesh."""

    min_point: Point
    max_point: Point

    def width(self) -> float:
        return float(self.max_point[0] - self.min_point[0])

    def height(self) -> float:
        return float(self.max_point[1] - self.min_point[1])


@final
class AdaptiveMeshCalculator:
    """Calculates adaptive mesh bounds and resolution."""

    def __init__(self, base_bounds: MeshBounds, base_resolution: tuple[int, int]):
        self.base_bounds = base_bounds
        self.base_resolution = base_resolution

    def calculate_adaptive_bounds(self, object_points: Sequence[Point], margin: float) -> MeshBounds:
        """Calculate adaptive bounds based on object points."""
        if not object_points:
            return self.base_bounds

        min_x = min(x for x, _ in object_points)
        max_x = max(x for x, _ in object_points)
        min_y = min(y for _, y in object_points)
        max_y = max(y for _, y in object_points)

        adapted_min_x = max(min_x - margin, self.base_bounds.min_point[0])
        adapted_max_x = min(max_x + margin, self.base_bounds.max_point[0])
        adapted_min_y = max(min_y - margin, self.base_bounds.min_point[1])
        adapted_max_y = min(max_y + margin, self.base_bounds.max_point[1])

        return MeshBounds(min_point=(adapted_min_x, adapted_min_y), max_point=(adapted_max_x, adapted_max_y))

    def calculate_adaptive_resolution(self, adaptive_bounds: MeshBounds) -> tuple[int, int]:
        """Calculate resolution maintaining point density."""
        base_width = self.base_bounds.width()
        base_height = self.base_bounds.height()

        if base_width == 0 or base_height == 0:
            return self.base_resolution

        x_density = (self.base_resolution[0] - 1) / base_width
        y_density = (self.base_resolution[1] - 1) / base_height

        adapted_width = adaptive_bounds.width()
        adapted_height = adaptive_bounds.height()

        # Grid resolution must be at least MIN_GRID_RESOLUTION x MIN_GRID_RESOLUTION
        x_res = max(MIN_GRID_RESOLUTION, ceil(adapted_width * x_density) + 1)
        y_res = max(MIN_GRID_RESOLUTION, ceil(adapted_height * y_density) + 1)

        return x_res, y_res
