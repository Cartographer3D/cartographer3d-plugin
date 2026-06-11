from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from typing_extensions import TypeAlias

from cartographer.macros.bed_mesh.paths.alternating_snake import AlternatingSnakePathGenerator
from cartographer.macros.bed_mesh.paths.random_path import RandomPathGenerator
from cartographer.macros.bed_mesh.paths.snake_path import SnakePathGenerator
from cartographer.macros.bed_mesh.paths.spiral_path import SpiralPathGenerator
from cartographer.macros.bed_mesh.paths.utils import apply_corner_radius_cap

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


# ---------------------------------------------------------------------------
# Corner-radius unit tests
# ---------------------------------------------------------------------------

# 3×3 grid at 20 mm spacing: x,y ∈ {0, 20, 40}.  With axis limits (-10, 50):
#   snake  auto-radius ≈ 9.5 mm  (bounded by min(0−(−10), 50−40) − 0.5 = 9.5)
#   spiral auto-radius = 1.0 mm  (hard cap at 1 mm; no doubling)
_CR_GRID: list[Point] = make_grid(3, 3, 20.0)
_CR_X_LIM = (-10.0, 50.0)
_CR_Y_LIM = (-10.0, 50.0)
_MESH_MAX_X = 40.0  # rightmost grid x


def _all_path_points_in_grid(path: list[Point], grid: list[Point], tol: float = 0.01) -> bool:
    """Return True iff every path point lies within *tol* of some grid point."""
    grid_arr = np.asarray(grid)
    return all(np.min(np.linalg.norm(grid_arr - np.asarray(pt), axis=1)) <= tol for pt in path)


class TestMaxCornerRadius:
    """mesh_max_corner_radius / MAX_CORNER_RADIUS semantics for path generators."""

    # --- SnakePathGenerator ---

    def test_snake_default_generates_arc_points(self):
        gen = SnakePathGenerator("x", max_corner_radius=None)
        path = list(gen.generate_path(_CR_GRID, _CR_X_LIM, _CR_Y_LIM))
        # auto-radius ~9.5 mm → u-turn arcs add extra off-grid points
        assert len(path) > len(_CR_GRID), "auto radius should generate arc points"
        assert not _all_path_points_in_grid(path, _CR_GRID), "arcs should produce off-grid points"

    def test_snake_zero_radius_no_arcs(self):
        gen = SnakePathGenerator("x", max_corner_radius=0.0)
        path = list(gen.generate_path(_CR_GRID, _CR_X_LIM, _CR_Y_LIM))
        assert _all_path_points_in_grid(path, _CR_GRID), "all points should be grid points (duplicates allowed)"

    def test_snake_positive_cap_limits_overshoot(self):
        cap = 2.0
        gen = SnakePathGenerator("x", max_corner_radius=cap)
        path = list(gen.generate_path(_CR_GRID, _CR_X_LIM, _CR_Y_LIM))
        # arcs still exist
        assert len(path) > len(_CR_GRID), "positive cap should still generate arc points"
        # u-turn arcs for x-direction snake extend along x beyond mesh_max_x by ~radius
        max_x = max(float(pt[0]) for pt in path)
        assert max_x <= _MESH_MAX_X + cap + 0.1, f"x overshoot {max_x:.3f} exceeds cap {cap} + mesh_max {_MESH_MAX_X}"
        # auto-radius would reach ~49.5; ensure cap actually constrained it
        auto_max_x = _MESH_MAX_X + 9.5
        assert max_x < auto_max_x - 1.0, (
            f"cap {cap} did not constrain overshoot: {max_x:.3f} close to auto {auto_max_x:.3f}"
        )

    # --- AlternatingSnakePathGenerator ---

    def test_alternating_snake_zero_radius_no_arcs(self):
        gen = AlternatingSnakePathGenerator("x", max_corner_radius=0.0)
        path = list(gen.generate_path(_CR_GRID, _CR_X_LIM, _CR_Y_LIM))
        assert _all_path_points_in_grid(path, _CR_GRID), "alternating snake with radius=0 should yield only grid points"

    def test_alternating_snake_default_generates_arc_points(self):
        gen = AlternatingSnakePathGenerator("x", max_corner_radius=None)
        path = list(gen.generate_path(_CR_GRID, _CR_X_LIM, _CR_Y_LIM))
        assert not _all_path_points_in_grid(path, _CR_GRID), (
            "alternating snake with auto radius should produce arc points"
        )

    # --- SpiralPathGenerator ---

    def test_spiral_default_generates_arc_points(self):
        gen = SpiralPathGenerator("x", max_corner_radius=None)
        path = list(gen.generate_path(_CR_GRID, _CR_X_LIM, _CR_Y_LIM))
        assert len(path) > len(_CR_GRID), "spiral auto radius should generate overshoot points"
        assert not _all_path_points_in_grid(path, _CR_GRID), "spiral arcs should produce off-grid points"

    def test_spiral_zero_radius_no_arcs(self):
        gen = SpiralPathGenerator("x", max_corner_radius=0.0)
        path = list(gen.generate_path(_CR_GRID, _CR_X_LIM, _CR_Y_LIM))
        assert _all_path_points_in_grid(path, _CR_GRID), "spiral with radius=0 should yield only grid points"

    def test_spiral_positive_cap_limits_overshoot(self):
        # spiral auto-radius = 1.0; cap at 0.5 should produce smaller arcs
        cap = 0.5
        gen_capped = SpiralPathGenerator("x", max_corner_radius=cap)
        gen_auto = SpiralPathGenerator("x", max_corner_radius=None)
        path_capped = list(gen_capped.generate_path(_CR_GRID, _CR_X_LIM, _CR_Y_LIM))
        path_auto = list(gen_auto.generate_path(_CR_GRID, _CR_X_LIM, _CR_Y_LIM))
        # Both should have more points than the grid (arcs exist)
        assert len(path_capped) > len(_CR_GRID)
        # Capped path should deviate less from the grid than auto
        grid_arr = np.asarray(_CR_GRID)
        max_dev_capped = max(float(np.min(np.linalg.norm(grid_arr - np.asarray(pt), axis=1))) for pt in path_capped)
        max_dev_auto = max(float(np.min(np.linalg.norm(grid_arr - np.asarray(pt), axis=1))) for pt in path_auto)
        assert max_dev_capped < max_dev_auto, (
            f"capped deviation {max_dev_capped:.3f} should be less than auto {max_dev_auto:.3f}"
        )


class TestApplyCornerRadiusCap:
    """Unit tests for the apply_corner_radius_cap shared helper."""

    def test_none_returns_auto(self):
        assert apply_corner_radius_cap(3.5, None) == 3.5

    def test_zero_disables_arcs(self):
        assert apply_corner_radius_cap(3.5, 0.0) == 0.0

    def test_positive_cap_clamps(self):
        assert apply_corner_radius_cap(3.5, 2.0) == 2.0

    def test_positive_cap_does_not_upsize(self):
        # cap larger than auto → auto wins
        assert apply_corner_radius_cap(1.0, 5.0) == 1.0

    def test_negative_auto_clamped_to_zero(self):
        # auto can transiently go negative (e.g. mesh touches axis limit)
        assert apply_corner_radius_cap(-0.1, None) == 0.0

    def test_negative_auto_with_cap(self):
        assert apply_corner_radius_cap(-1.0, 2.0) == 0.0
