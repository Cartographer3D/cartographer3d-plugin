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


@final
class SnakePathGenerator(PathGenerator):
    def __init__(self, main_direction: Literal["x", "y"], corner_radius: float):
        self.main_direction: Literal["x", "y"] = main_direction
        self.corner_radius = corner_radius

    @override
    def generate_path(self, points: list[Point]) -> Iterator[Point]:
        rows = cluster_points(points, self.main_direction)

        prev_row = rows[0]

        for i, row in enumerate(rows):
            row = list(row)
            if i % 2 == 1:
                row = row[::-1]

            if i > 0:
                # Create U-turn arc between previous end and current start
                prev_last = prev_row[-1]
                print(f"prev_last: {prev_last}, row: {row}, prev_row: {prev_row}")
                curr_first = row[0]
                entry_dir = row_direction(prev_row[-2:])

                yield from u_turn(prev_last, curr_first, entry_dir, self.corner_radius)

            yield from row
            prev_row = row


def u_turn(start: Point, end: Point, entry_dir: Vec, radius: float) -> Iterator[Point]:
    """Create two 90° arcs at each point for a smooth U-turn."""
    p1: Vec = np.array(start, dtype=float)
    p2: Vec = np.array(end, dtype=float)
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
