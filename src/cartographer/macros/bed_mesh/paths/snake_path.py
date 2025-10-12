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
    row_direction,
)

if TYPE_CHECKING:
    from cartographer.macros.bed_mesh.interfaces import Point

BUFFER = 0.5


@final
class SnakePathGenerator(PathGenerator):
    def __init__(self, main_direction: Literal["x", "y"]):
        self.main_direction: Literal["x", "y"] = main_direction

    @override
    def generate_path(
        self,
        points: list[Point],
        x_axis_limits: tuple[float, float],
        y_axis_limits: tuple[float, float],
    ) -> Iterator[Point]:
        grid = cluster_points(points, self.main_direction)

        axis_min, axis_max = x_axis_limits if self.main_direction == "x" else y_axis_limits
        main_index = 0 if self.main_direction == "x" else 1
        secondary_index = 1 if self.main_direction == "x" else 0

        row_spacing = abs(grid[1][0][secondary_index] - grid[0][0][secondary_index])
        max_radius_by_spacing = round(row_spacing / 2, 2)

        mesh_min = grid[0][0][main_index]
        mesh_max = grid[-1][-1][main_index]
        max_radius_by_bounds = min(mesh_min - axis_min, axis_max - mesh_max) - BUFFER

        # Final corner radius
        corner_radius = float(max(0, min(max_radius_by_spacing, max_radius_by_bounds)))

        prev_row = grid[0]

        for i, row in enumerate(grid):
            row = list(row)
            if i % 2 == 1:
                row.reverse()

            if i > 0:
                prev_last = prev_row[-1]
                curr_first = row[0]
                entry_dir = row_direction(prev_row[-2:])  # You must define this separately

                yield from u_turn(prev_last, curr_first, entry_dir, corner_radius)

            yield from row
            prev_row = row


def u_turn(start: Point, end: Point, entry_dir: Vec, radius: float) -> Iterator[Point]:
    """Create two 90° arcs at each point for a smooth U-turn."""
    p1: Vec = np.asarray(start, dtype=float)
    p2: Vec = np.asarray(end, dtype=float)
    delta = p2 - p1

    if np.linalg.norm(delta) == 0:
        return  # skip zero-distance turn

    turn_dir = normalize(delta)

    # Determine if the turn is CCW or CW based on 2D cross product
    # We'll assume a horizontal travel: if moving left to right, then back right to left
    # Then the perpendicular should flip for the reverse direction
    turn_ccw = bool(np.cross(entry_dir, turn_dir) > 0)  # flip depending on entry direction
    turn_angle = 90 if turn_ccw else -90

    entry_perp = perpendicular(entry_dir, ccw=turn_ccw)
    start_angle = angle_deg(-entry_perp)

    offset = entry_perp * radius
    yield from arc_points(p1 + offset, radius, start_angle, turn_angle)
    yield from arc_points(p2 - offset, radius, start_angle + turn_angle, turn_angle)
