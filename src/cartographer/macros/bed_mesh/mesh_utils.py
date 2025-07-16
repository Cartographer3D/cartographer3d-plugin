from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal

import numpy as np

from cartographer.interfaces.printer import Position

if TYPE_CHECKING:
    from cartographer.interfaces.printer import Sample
    from cartographer.macros.bed_mesh.interfaces import Point

logger = logging.getLogger(__name__)


def cluster_points(points: list[Point], axis: Literal["x", "y"], tol: float = 1e-3) -> list[list[Point]]:
    # axis to cluster on:
    # if main_direction = "x", cluster on y (index 1)
    # if main_direction = "y", cluster on x (index 0)
    cluster_index = 1 if axis == "x" else 0
    sort_index = 0 if axis == "x" else 1

    clusters: dict[float, list[Point]] = defaultdict(list)
    for p in points:
        key = round(p[cluster_index] / tol)
        clusters[key].append(p)

    sorted_keys = sorted(clusters.keys())

    rows: list[list[Point]] = []
    for key in sorted_keys:
        row_points = clusters[key]
        row_points.sort(key=lambda pt: pt[sort_index])
        rows.append(row_points)

    return rows


@dataclass(frozen=True)
class GridPointResult:
    point: Point
    z: float
    sample_count: int


def assign_samples_to_grid(
    grid: list[Point], samples: list[Sample], calculate_height: Callable[[Sample], float], max_distance: float = 1.0
) -> list[GridPointResult]:
    # Extract sorted unique coordinates
    mesh_array: np.ndarray[float, np.dtype[np.float64]] = np.array(grid)
    x_vals = np.unique(mesh_array[:, 0])
    y_vals = np.unique(mesh_array[:, 1])

    x_res = len(x_vals)
    y_res = len(y_vals)

    x_min, x_max = float(x_vals[0]), float(x_vals[-1])
    y_min, y_max = float(y_vals[0]), float(y_vals[-1])

    x_step: float = (x_max - x_min) / (x_res - 1)
    y_step: float = (y_max - y_min) / (y_res - 1)

    # Map (j, i) grid positions to (x, y)
    index_to_point: dict[tuple[int, int], Point] = {
        (j, i): (float(x), float(y)) for i, x in enumerate(x_vals) for j, y in enumerate(y_vals)
    }

    # Accumulator: (row=j, col=i) → list of z values
    accumulator: dict[tuple[int, int], list[float]] = defaultdict(list)

    for sample in samples:
        if sample.position is None:
            continue
        sx = sample.position.x
        sy = sample.position.y
        i = round((sx - x_min) / x_step)
        j = round((sy - y_min) / y_step)

        if 0 <= i < x_res and 0 <= j < y_res:
            gx: float = x_vals[i]
            gy: float = y_vals[j]
            if np.hypot(sx - gx, sy - gy) > max_distance:
                continue

            sz = calculate_height(sample)
            accumulator[(j, i)].append(sz)

    results: list[GridPointResult] = []

    for (j, i), point in index_to_point.items():
        values = accumulator.get((j, i), [])
        count = len(values)
        z = float(np.median(values))
        results.append(GridPointResult(point=point, z=z, sample_count=count))

    return results


def normalize_to_zero_reference(
    positions: list[Position],
    zero_ref: Point,
) -> list[Position]:
    # Step 1: Extract sorted unique coordinates
    xs = np.array(sorted({p.x for p in positions}))
    ys = np.array(sorted({p.y for p in positions}))

    # Step 2: Build a mapping from (x, y) to z
    z_map = {(p.x, p.y): p.z for p in positions}

    # Step 3: Build Z grid with shape (y_count, x_count)
    z_grid = np.array([[z_map[(x, y)] for x in xs] for y in ys])

    # Step 4: Compute grid spacing
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]

    # Step 5: Bilinear interpolation
    tx = (zero_ref[0] - xs[0]) / dx
    ty = (zero_ref[1] - ys[0]) / dy
    xi = int(np.floor(tx))
    yi = int(np.floor(ty))
    tx -= xi
    ty -= yi

    z00 = z_grid[yi, xi]
    z01 = z_grid[yi, xi + 1]
    z10 = z_grid[yi + 1, xi]
    z11 = z_grid[yi + 1, xi + 1]

    z0 = (1 - tx) * z00 + tx * z01
    z1 = (1 - tx) * z10 + tx * z11
    z_ref = (1 - ty) * z0 + ty * z1

    logger.debug("Interpolated zero reference at (%.2f, %.2f) = %.3f", zero_ref[0], zero_ref[1], z_ref)

    # Step 6: Normalize
    z_grid -= z_ref

    # Step 7: Rebuild list in y-major order
    normalized_positions = [
        Position(x=float(x), y=float(y), z=float(z_grid[yi, xi])) for yi, y in enumerate(ys) for xi, x in enumerate(xs)
    ]
    return normalized_positions
