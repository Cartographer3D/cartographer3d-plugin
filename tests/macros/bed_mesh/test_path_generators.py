from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from typing_extensions import TypeAlias

from cartographer.macros.bed_mesh.paths.random_path import RandomPathGenerator
from cartographer.macros.bed_mesh.paths.snake_path import SnakePathGenerator
from cartographer.macros.bed_mesh.paths.spiral_path import SpiralPathGenerator

if TYPE_CHECKING:
    from cartographer.macros.bed_mesh.interfaces import PathGenerator, Point


def make_grid(nx: int, ny: int, spacing: float) -> list[Point]:
    return [(x * spacing, y * spacing) for y in range(ny) for x in range(nx)]


GeneratorFixture: TypeAlias = "tuple[str, PathGenerator]"
GridFixture: TypeAlias = "tuple[str, list[Point]]"


@pytest.fixture(
    params=[
        ("Snake X", lambda: SnakePathGenerator(main_direction="x")),
        ("Snake Y", lambda: SnakePathGenerator(main_direction="y")),
        ("Spiral", lambda: SpiralPathGenerator(main_direction="x")),
        ("Random", lambda: RandomPathGenerator(main_direction="x")),
    ]
)
def generator(request: pytest.FixtureRequest):
    name, gen_factory = request.param
    return name, gen_factory()


@pytest.fixture(
    params=[
        ("3x3 grid", make_grid(3, 3, 1.0)),
        ("3x5 grid", make_grid(3, 5, 1.0)),
        ("4x4 grid", make_grid(4, 4, 1.0)),
        ("5x3 grid", make_grid(5, 3, 1.0)),
        ("7x8 grid", make_grid(7, 8, 1.0)),
        ("8x7 grid", make_grid(8, 7, 1.0)),
    ]
)
def grid_points(request: pytest.FixtureRequest):
    return request.param


def test_path_generator_covers_all_points(generator: GeneratorFixture, grid_points: GridFixture):
    gen_name, gen = generator
    grid_name, points = grid_points

    max_dist = 0.2  # maximum allowed miss distance per input point

    path = list(gen.generate_path(points, (0, 100), (0, 100)))
    path_array = np.asarray(path)

    for _, pt in enumerate(points):
        pt = np.asarray(pt)
        dists = np.linalg.norm(path_array - pt, axis=1)
        min_dist = np.min(dists)
        assert min_dist <= max_dist, (
            f"{gen_name} did not reach input point {pt.tolist()} (min dist {min_dist:.3f} > {max_dist}) on {grid_name}"
        )

    # Optional: continuity check
    max_step = 10.0
    for p0, p1 in zip(path, path[1:]):
        dist = np.linalg.norm(np.asarray(p1) - np.asarray(p0))
        assert dist <= max_step, f"{gen_name} discontinuity {dist:.2f} on {grid_name}"
