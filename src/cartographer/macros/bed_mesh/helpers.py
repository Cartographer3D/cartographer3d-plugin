from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cached_property
from math import ceil
from typing import TYPE_CHECKING, Callable, Sequence, final, overload

import numpy as np

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

    def __post_init__(self):
        if self.x_resolution < MIN_GRID_RESOLUTION or self.y_resolution < MIN_GRID_RESOLUTION:
            msg = f"Grid resolution must be at least {MIN_GRID_RESOLUTION}x{MIN_GRID_RESOLUTION}"
            raise ValueError(msg)

    # cached_property requires __dict__; do not add __slots__ to MeshGrid.
    @cached_property
    def x_coords(self) -> NDArray[np.float_]:
        """Get array of x coordinates."""
        arr = np.round(np.linspace(self.min_point[0], self.max_point[0], self.x_resolution), 2)
        arr.setflags(write=False)
        return arr

    @cached_property
    def y_coords(self) -> NDArray[np.float_]:
        """Get array of y coordinates."""
        arr = np.round(np.linspace(self.min_point[1], self.max_point[1], self.y_resolution), 2)
        arr.setflags(write=False)
        return arr

    @property
    def x_step(self) -> float:
        """Get step size in x direction."""
        return float(self.max_point[0] - self.min_point[0]) / (self.x_resolution - 1)

    @property
    def y_step(self) -> float:
        """Get step size in y direction."""
        return float(self.max_point[1] - self.min_point[1]) / (self.y_resolution - 1)

    def generate_points(self) -> list[Point]:
        """Generate all grid points in y-major order."""
        return [(float(x), float(y)) for x in self.x_coords for y in self.y_coords]

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
        x_res = self.grid.x_resolution
        y_res = self.grid.y_resolution
        z_out: NDArray[np.float64] = np.full(y_res * x_res, np.nan, dtype=np.float64)
        c_out: NDArray[np.intp] = np.zeros(y_res * x_res, dtype=np.intp)

        selected_samples, cell_ids = self._select_grid_samples(samples)
        if selected_samples:
            heights: NDArray[np.float64] = np.fromiter(
                (calculate_height(sample) for sample in selected_samples),
                dtype=np.float64,
                count=len(selected_samples),
            )

            order: NDArray[np.intp] = np.lexsort((heights, cell_ids))
            sorted_cells = cell_ids[order]
            sorted_heights = heights[order]

            unique_cells, starts, counts = np.unique(sorted_cells, return_index=True, return_counts=True)
            ends = starts + counts
            mid_lo = starts + (counts - 1) // 2
            mid_hi = starts + counts // 2
            lo_vals = sorted_heights[mid_lo]
            hi_vals = sorted_heights[mid_hi]
            with np.errstate(invalid="ignore"):
                medians: NDArray[np.float64] = np.where(mid_lo == mid_hi, lo_vals, 0.5 * (lo_vals + hi_vals))
            medians[np.isnan(sorted_heights[ends - 1])] = np.nan

            z_out[unique_cells] = medians
            c_out[unique_cells] = counts

        x_coords = self.grid.x_coords
        y_coords = self.grid.y_coords
        z_grid = z_out.reshape(y_res, x_res)
        c_grid = c_out.reshape(y_res, x_res)
        return [
            GridPointResult(
                point=(float(x_coords[i]), float(y_coords[j])),
                z=float(z_grid[j, i]),
                sample_count=int(c_grid[j, i]),
            )
            for j in range(y_res)
            for i in range(x_res)
        ]

    def _select_grid_samples(self, samples: list[Sample]) -> tuple[list[Sample], NDArray[np.intp]]:
        """Filter samples to those within grid bounds and assignment radius, returning matching flat cell indices."""
        positioned_samples: list[Sample] = []
        coords: list[tuple[float, float]] = []
        for sample in samples:
            pos = sample.position
            if pos is not None:
                positioned_samples.append(sample)
                coords.append((pos.x, pos.y))

        empty_cells: NDArray[np.intp] = np.empty(0, dtype=np.intp)
        if not positioned_samples:
            return [], empty_cells

        xy: NDArray[np.float64] = np.asarray(coords, dtype=np.float64)
        xs = xy[:, 0]
        ys = xy[:, 1]
        min_x, min_y = self.grid.min_point
        max_x, max_y = self.grid.max_point
        eps = self.max_distance
        in_bounds = (xs >= min_x - eps) & (xs <= max_x + eps) & (ys >= min_y - eps) & (ys <= max_y + eps)
        bound_indices: NDArray[np.intp] = np.flatnonzero(in_bounds)
        if bound_indices.size == 0:
            return [], empty_cells

        if self.grid.x_step == 0.0 or self.grid.y_step == 0.0:
            msg = "Grid step size must be non-zero"
            raise ZeroDivisionError(msg)

        bx = xs[bound_indices]
        by = ys[bound_indices]
        ii: NDArray[np.intp] = np.rint((bx - min_x) / self.grid.x_step).astype(np.intp)
        jj: NDArray[np.intp] = np.rint((by - min_y) / self.grid.y_step).astype(np.intp)
        valid_idx = (ii >= 0) & (ii < self.grid.x_resolution) & (jj >= 0) & (jj < self.grid.y_resolution)

        bound_indices = bound_indices[valid_idx]
        bx = bx[valid_idx]
        by = by[valid_idx]
        ii = ii[valid_idx]
        jj = jj[valid_idx]

        distances: NDArray[np.float64] = np.hypot(bx - self.grid.x_coords[ii], by - self.grid.y_coords[jj])
        in_radius = distances <= self.max_distance

        keep_indices = bound_indices[in_radius]
        cell_ids = jj[in_radius] * self.grid.x_resolution + ii[in_radius]
        selected_samples = [positioned_samples[int(idx)] for idx in keep_indices]
        return selected_samples, cell_ids


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
        """Normalize positions to a zero reference point using bilinear interpolation."""
        if not positions:
            return []
        # Convert to NumPy array (shape: (N, 3))
        arr: NDArray[np.float_] = np.asarray([(p.x, p.y, p.z) for p in positions])

        # Extract sorted unique coords
        xs = np.unique(arr[:, 0])
        ys = np.unique(arr[:, 1])

        # Sort positions in matching y-major order and reshape z
        # This works if positions form a complete rectangular grid
        # and can be sorted by (y, x)
        sort_idx = np.lexsort((arr[:, 0], arr[:, 1]))
        z_grid = arr[sort_idx, 2].reshape(len(ys), len(xs))

        # Determine zero reference height
        if zero_height is not None:
            z_ref = zero_height
        elif zero_ref is not None:
            z_ref = self._bilinear_interpolate(xs, ys, z_grid, zero_ref)
        else:
            msg = "Either zero_ref or zero_height must be provided."
            raise ValueError(msg)

        # Normalize
        z_grid -= z_ref

        # Rebuild output in y-major order
        normalized_positions = [
            Position(x=float(x), y=float(y), z=float(z)) for y, row in zip(ys, z_grid) for x, z in zip(xs, row)
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

        # Sort into y-major order and reshape
        sort_idx = np.lexsort((arr[:, 0], arr[:, 1]))
        z_grid = arr[sort_idx, 2].reshape(len(ys), len(xs))

        # Build mask (ys major, xs minor)
        mask: NDArray[np.bool_] = np.zeros_like(z_grid, dtype=bool)
        for i in range(len(ys)):
            for j in range(len(xs)):
                point = (xs[j], ys[i])
                if any(region.contains_point(point) for region in faulty_regions):
                    mask[i, j] = True

        z_grid_masked = np.where(mask, np.nan, z_grid)

        # Interpolate missing values if scipy is available
        if np.any(mask):
            scipy_helpers.raise_if_rbf_interpolator_unavailable()
            logger.info("Interpolating %d faulty points", np.sum(mask))

            # Use the same order as flattening the mask/grid
            X, Y = np.meshgrid(xs, ys)  # noqa: N806
            points_grid = np.column_stack([X.ravel(), Y.ravel()])

            valid_points = points_grid[~mask.ravel()]
            valid_values = z_grid_masked[~mask]

            rbf = rbf_interpolator(valid_points, valid_values, neighbors=64, smoothing=0.0)

            missing_points = points_grid[mask.ravel()]
            interpolated = rbf(missing_points)  # pyright: ignore[reportUnknownVariableType]

            z_grid_masked[mask] = interpolated

        new_positions = [
            Position(x=float(x), y=float(y), z=float(z)) for y, row in zip(ys, z_grid_masked) for x, z in zip(xs, row)
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
