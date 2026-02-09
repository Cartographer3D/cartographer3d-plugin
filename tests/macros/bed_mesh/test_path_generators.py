from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from typing_extensions import TypeAlias

from cartographer.macros.bed_mesh.paths.random_path import RandomPathGenerator
from cartographer.macros.bed_mesh.paths.snake_path import SnakePathGenerator
from cartographer.macros.bed_mesh.paths.spiral_path import SpiralPathGenerator
from cartographer.macros.bed_mesh.paths.circular_snake_path import CircularSnakePathGenerator

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


class TestCircularSnakePathGenerator:
    """Specific tests for circular snake path generator.
       Simple test for now, this could be enchanced later"""

    def test_circular_snake_alternates_row_directions(self):
        """Test circular snake path with realistic sparse grid pattern."""
        gen = CircularSnakePathGenerator(main_direction="x")

        # Realistic circular mesh: 5x5 grid with radius=30, origin=(50,50), grid 30-70 with 10mm spacing
        points: list[Point] = [
            # Row y=30: 3 points
            (40.0, 30.0),
            (50.0, 30.0),
            (60.0, 30.0),
            # Row y=40: 5 points
            (30.0, 40.0),
            (40.0, 40.0),
            (50.0, 40.0),
            (60.0, 40.0),
            (70.0, 40.0),
            # Row y=50: 5 points (center)
            (30.0, 50.0),
            (40.0, 50.0),
            (50.0, 50.0),
            (60.0, 50.0),
            (70.0, 50.0),
            # Row y=60: 5 points
            (30.0, 60.0),
            (40.0, 60.0),
            (50.0, 60.0),
            (60.0, 60.0),
            (70.0, 60.0),
            # Row y=70: 3 points
            (40.0, 70.0),
            (50.0, 70.0),
            (60.0, 70.0),
        ]

        path = list(gen.generate_path(points, (30.0, 70.0), (30.0, 70.0)))
        path_array = np.asarray(path)

        assert len(path) == len(points)
        max_dist = 0.01  

        for pt in points:
            pt_array = np.asarray(pt)
            dists = np.linalg.norm(path_array - pt_array, axis=1)
            min_dist = np.min(dists)
            assert min_dist <= max_dist, f"Point {pt} not reached (min dist {min_dist:.3f})"

        max_step = 50.0 
        for p0, p1 in zip(path, path[1:]):
            dist = np.linalg.norm(np.asarray(p1) - np.asarray(p0))
            assert dist <= max_step, f"Discontinuity {dist:.2f} between {p0} and {p1}"

        # Verify snake pattern: check that row 1 and 3 are reversed (rightmost point first)
        # Row 1 should start at x=70
        assert path[3][0] == 70.0, "Row 1 should be reversed (start at x=70)"
        # Row 3 should start at x=70
        assert path[13][0] == 70.0, "Row 3 should be reversed (start at x=70)"
