from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, final

from typing_extensions import override

from cartographer.macros.bed_mesh.interfaces import PathGenerator
from cartographer.macros.bed_mesh.paths.snake_path import SnakePathGenerator

if TYPE_CHECKING:
    from cartographer.macros.bed_mesh.interfaces import Point


@final
class AlternatingSnakePathGenerator(PathGenerator):
    def __init__(self, main_direction: str, max_corner_radius: float | None = None):
        self.main_direction: str = main_direction
        self.max_corner_radius: float | None = max_corner_radius

    @override
    def generate_path(
        self,
        points: list[Point],
        x_axis_limits: tuple[float, float],
        y_axis_limits: tuple[float, float],
    ) -> Iterator[Point]:
        alternate_direction = "y" if self.main_direction == "x" else "x"
        main_path = SnakePathGenerator(self.main_direction, self.max_corner_radius)
        alternate_path = SnakePathGenerator(alternate_direction, self.max_corner_radius)
        yield from main_path.generate_path(points, x_axis_limits, y_axis_limits)
        yield from reversed(list(alternate_path.generate_path(points, x_axis_limits, y_axis_limits)))
