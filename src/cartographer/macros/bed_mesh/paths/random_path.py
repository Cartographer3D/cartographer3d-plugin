from __future__ import annotations

import random
from typing import TYPE_CHECKING, Iterator, Literal, final

import numpy as np
from typing_extensions import override

from cartographer.macros.bed_mesh.interfaces import PathGenerator

if TYPE_CHECKING:
    from cartographer.macros.bed_mesh.interfaces import Point

MIN_DIST = 10.0


@final
class RandomPathGenerator(PathGenerator):
    def __init__(self, main_direction: Literal["x", "y"]):
        del main_direction

    @override
    def generate_path(
        self,
        points: list[Point],
        x_axis_limits: tuple[float, float],
        y_axis_limits: tuple[float, float],
    ) -> Iterator[Point]:
        del x_axis_limits, y_axis_limits

        if not points:
            return iter([])

        remaining = points[:]
        current = random.choice(remaining)
        yield current
        remaining.remove(current)

        while remaining:
            current_arr = np.asarray(current)
            dists = np.asarray([np.linalg.norm(np.asarray(p) - current_arr) for p in remaining])

            if np.all(dists == 0):
                next_point = random.choice(remaining)
            else:
                weights = dists / dists.sum()
                next_point = random.choices(remaining, weights=weights, k=1)[0]

            yield next_point
            remaining.remove(next_point)
            current = next_point
