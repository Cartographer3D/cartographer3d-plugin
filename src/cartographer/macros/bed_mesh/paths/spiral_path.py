from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, Literal, final

import numpy as np
from typing_extensions import override

from cartographer.macros.bed_mesh.interfaces import PathGenerator
from cartographer.macros.bed_mesh.paths.utils import (
    Vec,
    angle_deg,
    arc_points,
    cluster_points,
    normalize,
    perpendicular,
)

if TYPE_CHECKING:
    from cartographer.macros.bed_mesh.interfaces import Point

BUFFER = 0.5


@final
class SpiralPathGenerator(PathGenerator):
    def __init__(self, main_direction: Literal["x", "y"]):
        del main_direction

    @override
    def generate_path(
        self,
        points: list[Point],
        x_axis_limits: tuple[float, float],
        y_axis_limits: tuple[float, float],
    ) -> Iterator[Point]:
        grid = cluster_points(points, axis="x")  # Bottom row is index 0
        rows = len(grid)
        cols = len(grid[0]) if rows else 0
        offset = 0

        mesh_min_x, mesh_min_y = grid[0][0]
        mesh_max_x, mesh_max_y = grid[-1][-1]
        max_radius_x = min(mesh_min_x - x_axis_limits[0], x_axis_limits[1] - mesh_max_x) - BUFFER
        max_radius_y = min(mesh_min_y - y_axis_limits[0], y_axis_limits[1] - mesh_max_y) - BUFFER
        corner_radius = float(max(0, min(max_radius_x, max_radius_y, 1))) * 2
        print(corner_radius)

        while offset < (rows + 1) // 2 and offset < (cols + 1) // 2:
            bottom = grid[offset][offset : cols - offset]
            right = [grid[i][cols - offset - 1] for i in range(offset + 1, rows - offset)]
            top = grid[rows - offset - 1][offset : cols - offset - 1][::-1] if rows - offset - 1 != offset else []
            left = (
                [grid[i][offset] for i in range(rows - offset - 2, offset, -1)] if cols - offset - 1 != offset else []
            )

            # === Bottom row (→)
            if right or top or left:
                yield from bottom[:-1]
                yield from corner(bottom[-1], (1.0, 0.0), corner_radius)
            else:
                yield from bottom  # Last leg, include final point

            # === Right column (↑)
            if top or left:
                yield from right[:-1]
                yield from corner(right[-1], (0.0, 1.0), corner_radius)
            else:
                yield from right

            # === Top row (←)
            if left:
                yield from top[:-1]
                yield from corner(top[-1], (-1.0, 0.0), corner_radius)
            else:
                yield from top

            # === Left column (↓)
            if left:
                if offset + 1 < (rows + 1) // 2 and offset + 1 < (cols + 1) // 2:
                    # There will be another ring → add corner
                    if left:
                        yield from left[:-1]
                        yield from corner(left[-1], (0.0, -1.0), corner_radius)
                else:
                    # Final leg
                    yield from left

            offset += 1


def corner(point: Point, entry_dir: tuple[float, float], radius: float) -> Iterator[Point]:
    p1: Vec = np.asarray(point, dtype=float)
    direction: Vec = np.asarray(entry_dir, dtype=float)
    turn_ccw = True
    turn_angle = 90

    entry_perp = perpendicular(direction, ccw=turn_ccw)
    between = normalize(entry_perp - direction)
    start_angle = angle_deg(-between) - turn_angle / 2

    offset = between * radius
    yield from arc_points(p1 + offset, radius, start_angle, turn_angle)
