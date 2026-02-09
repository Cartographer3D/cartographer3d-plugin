from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, Literal, final
from typing_extensions import override

from cartographer.macros.bed_mesh.interfaces import PathGenerator
from cartographer.macros.bed_mesh.paths.utils import cluster_points

if TYPE_CHECKING:
    from cartographer.macros.bed_mesh.interfaces import Point


@final
class CircularSnakePathGenerator(PathGenerator):
    """Simple path generator for circular/sparse mesh grids.
    
    Creates a basic snake pattern by alternating row directions
    for efficient probe movement through variable-length rows.
    """

    def __init__(self, main_direction: Literal["x", "y"]):
        self.main_direction: Literal["x", "y"] = main_direction

    @override
    def generate_path(
        self,
        points: list[Point],
        x_axis_limits: tuple[float, float],
        y_axis_limits: tuple[float, float],
    ) -> Iterator[Point]:
        """Generate a simple snake path through sparse circular mesh points."""
        if not points:
            return

        grid = cluster_points(points, self.main_direction)

        for i, row in enumerate(grid):
            row = list(row)
            # Alternate row directions for snake pattern
            if i % 2 == 1:
                row.reverse()
                
            yield from row


