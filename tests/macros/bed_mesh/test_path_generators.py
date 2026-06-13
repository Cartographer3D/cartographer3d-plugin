from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from typing_extensions import TypeAlias

from cartographer.macros.bed_mesh.paths.alternating_snake import AlternatingSnakePathGenerator
from cartographer.macros.bed_mesh.paths.hilbert_path import HilbertPathGenerator, next_power_of_two, xy_to_hilbert
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
        ("Alternating Snake X", lambda: AlternatingSnakePathGenerator(main_direction="x")),
        ("Alternating Snake Y", lambda: AlternatingSnakePathGenerator(main_direction="y")),
        ("Spiral", lambda: SpiralPathGenerator(main_direction="x")),
        ("Random", lambda: RandomPathGenerator(main_direction="x")),
        ("Hilbert X", lambda: HilbertPathGenerator(main_direction="x")),
        ("Hilbert Y", lambda: HilbertPathGenerator(main_direction="y")),
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


# ---------------------------------------------------------------------------
# Hilbert path internals
# ---------------------------------------------------------------------------


class TestNextPowerOfTwo:
    def test_one(self):
        assert next_power_of_two(1) == 1

    def test_already_power_of_two(self):
        assert next_power_of_two(4) == 4
        assert next_power_of_two(8) == 8
        assert next_power_of_two(16) == 16

    def test_rounds_up(self):
        assert next_power_of_two(3) == 4
        assert next_power_of_two(5) == 8
        assert next_power_of_two(10) == 16
        assert next_power_of_two(11) == 16
        assert next_power_of_two(20) == 32
        assert next_power_of_two(21) == 32

    def test_zero_or_negative_returns_one(self):
        assert next_power_of_two(0) == 1
        assert next_power_of_two(-5) == 1


class TestXyToHilbert:
    """Spot-check xy_to_hilbert for small power-of-two grids."""

    def test_2x2_all_indices_unique(self):
        """All four (x,y) positions in a 2×2 square get distinct distances."""
        distances = {xy_to_hilbert(2, x, y) for x in range(2) for y in range(2)}
        assert distances == {0, 1, 2, 3}

    def test_4x4_all_indices_unique(self):
        distances = {xy_to_hilbert(4, x, y) for x in range(4) for y in range(4)}
        assert len(distances) == 16
        assert distances == set(range(16))

    def test_origin_is_always_zero(self):
        for order in (1, 2, 4, 8, 16):
            assert xy_to_hilbert(order, 0, 0) == 0


# ---------------------------------------------------------------------------
# HilbertPathGenerator behaviour
# ---------------------------------------------------------------------------


_AXIS_LIMITS = (-10.0, 300.0)


class TestHilbertPathGeneratorExact:
    """Exact output tests for small, well-known grids."""

    def test_1x1_grid_yields_single_point(self):
        gen = HilbertPathGenerator("x")
        pts: list[Point] = [(5.0, 7.0)]
        path = list(gen.generate_path(pts, _AXIS_LIMITS, _AXIS_LIMITS))
        assert path == [(5.0, 7.0)]

    def test_empty_grid_yields_nothing(self):
        gen = HilbertPathGenerator("x")
        path = list(gen.generate_path([], _AXIS_LIMITS, _AXIS_LIMITS))
        assert path == []

    def test_2x2_x_direction_exact_order(self):
        """For a 2×2 unit grid, main_direction='x' first step is +X."""
        # grid[0] = [(0,0),(1,0)], grid[1] = [(0,1),(1,1)]
        # r=0,c=0 → hilbert(2,0,0)=0  → (0,0)
        # r=0,c=1 → hilbert(2,0,1)=1  → (1,0)
        # r=1,c=1 → hilbert(2,1,1)=2  → (1,1)
        # r=1,c=0 → hilbert(2,1,0)=3  → (0,1)
        gen = HilbertPathGenerator("x")
        pts: list[Point] = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
        path = list(gen.generate_path(pts, _AXIS_LIMITS, _AXIS_LIMITS))
        assert path[0] == (0.0, 0.0), "first point must be origin"
        # First move should be in the +X direction
        assert float(path[1][0]) > float(path[0][0]), "first step should be +X for main_direction='x'"
        assert len(path) == 4

    def test_2x2_y_direction_exact_order(self):
        """For a 2×2 unit grid, main_direction='y' first step is +Y."""
        gen = HilbertPathGenerator("y")
        pts: list[Point] = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
        path = list(gen.generate_path(pts, _AXIS_LIMITS, _AXIS_LIMITS))
        assert path[0] == (0.0, 0.0), "first point must be origin"
        # First move should be in the +Y direction
        assert float(path[1][1]) > float(path[0][1]), "first step should be +Y for main_direction='y'"
        assert len(path) == 4

    def test_x_and_y_directions_differ(self):
        """x and y orientation produce different orderings on an asymmetric grid."""
        pts = make_grid(3, 4, 1.0)  # rectangular → orientations differ
        gen_x = HilbertPathGenerator("x")
        gen_y = HilbertPathGenerator("y")
        path_x = list(gen_x.generate_path(pts, _AXIS_LIMITS, _AXIS_LIMITS))
        path_y = list(gen_y.generate_path(pts, _AXIS_LIMITS, _AXIS_LIMITS))
        assert path_x != path_y, "x and y orientations must produce different traversal orders"


class TestHilbertPathGeneratorNoDuplicates:
    """Each input mesh point is visited exactly once."""

    @pytest.mark.parametrize(
        "nx,ny",
        [
            (3, 3),
            (4, 4),
            (5, 5),
            (10, 10),
            (11, 11),
            (20, 20),
            (21, 21),
            (3, 5),
            (5, 3),
            (4, 7),
            (7, 4),
        ],
    )
    def test_no_duplicates_covers_all_points(self, nx: int, ny: int):
        pts = make_grid(nx, ny, 1.0)
        gen = HilbertPathGenerator("x")
        path = list(gen.generate_path(pts, _AXIS_LIMITS, _AXIS_LIMITS))
        assert len(path) == len(pts), f"expected {len(pts)} points for {nx}×{ny}, got {len(path)}"
        # Convert to rounded tuples for set comparison (floats may vary slightly)
        path_set = {(round(float(x), 6), round(float(y), 6)) for x, y in path}
        pts_set = {(round(float(x), 6), round(float(y), 6)) for x, y in pts}
        assert path_set == pts_set, f"{nx}×{ny}: path does not cover all input points exactly"


class TestHilbertPathGeneratorDeterministic:
    """Repeated calls with identical input produce identical output."""

    @pytest.mark.parametrize("direction", ["x", "y"])
    def test_deterministic(self, direction: str):
        pts = make_grid(5, 5, 2.5)
        gen = HilbertPathGenerator(direction)
        path_a = list(gen.generate_path(pts, _AXIS_LIMITS, _AXIS_LIMITS))
        path_b = list(gen.generate_path(pts, _AXIS_LIMITS, _AXIS_LIMITS))
        assert path_a == path_b, f"Hilbert {direction!r}: two calls produced different orderings"


class TestHilbertPathGeneratorRectangular:
    """Rectangular (non-square) grids are handled correctly."""

    @pytest.mark.parametrize(
        "nx,ny",
        [
            (3, 7),
            (7, 3),
            (1, 10),
            (10, 1),
            (6, 11),
        ],
    )
    def test_rectangular_covers_all_points(self, nx: int, ny: int):
        pts = make_grid(nx, ny, 1.0)
        gen = HilbertPathGenerator("x")
        path = list(gen.generate_path(pts, _AXIS_LIMITS, _AXIS_LIMITS))
        assert len(path) == nx * ny
        path_set = {(round(float(x), 6), round(float(y), 6)) for x, y in path}
        pts_set = {(round(float(x), 6), round(float(y), 6)) for x, y in pts}
        assert path_set == pts_set


class TestHilbertPathGeneratorMaxCornerRadius:
    """max_corner_radius is accepted but has no effect on the output."""

    def test_radius_none_same_as_zero(self):
        pts = make_grid(4, 4, 1.0)
        gen_none = HilbertPathGenerator("x", max_corner_radius=None)
        gen_zero = HilbertPathGenerator("x", max_corner_radius=0.0)
        gen_pos = HilbertPathGenerator("x", max_corner_radius=5.0)
        path_none = list(gen_none.generate_path(pts, _AXIS_LIMITS, _AXIS_LIMITS))
        path_zero = list(gen_zero.generate_path(pts, _AXIS_LIMITS, _AXIS_LIMITS))
        path_pos = list(gen_pos.generate_path(pts, _AXIS_LIMITS, _AXIS_LIMITS))
        # All three should produce identical paths (no arc points added)
        assert path_none == path_zero == path_pos
        assert len(path_none) == 16  # exactly the grid points, no extras


class TestHilbertPathRegistration:
    """MeshPath.HILBERT is registered in PATH_GENERATOR_MAP."""

    def test_hilbert_in_mesh_path_enum(self):
        from cartographer.interfaces.configuration import MeshPath

        assert MeshPath("hilbert") == MeshPath.HILBERT
        assert str(MeshPath.HILBERT) == "hilbert"

    def test_hilbert_in_path_generator_map(self):
        from cartographer.interfaces.configuration import MeshPath
        from cartographer.macros.bed_mesh.scan_mesh import PATH_GENERATOR_MAP

        assert MeshPath.HILBERT in PATH_GENERATOR_MAP
        assert PATH_GENERATOR_MAP[MeshPath.HILBERT] is HilbertPathGenerator

    def test_path_generator_map_produces_correct_instance(self):
        from cartographer.interfaces.configuration import MeshPath
        from cartographer.macros.bed_mesh.scan_mesh import PATH_GENERATOR_MAP

        gen = PATH_GENERATOR_MAP[MeshPath.HILBERT]("x", None)
        assert isinstance(gen, HilbertPathGenerator)


class TestAlternatingSnakePathRegistration:
    """MeshPath.ALTERNATING_SNAKE is registered in PATH_GENERATOR_MAP."""

    def test_alternating_snake_in_path_generator_map(self):
        from cartographer.interfaces.configuration import MeshPath
        from cartographer.macros.bed_mesh.scan_mesh import PATH_GENERATOR_MAP

        assert MeshPath.ALTERNATING_SNAKE in PATH_GENERATOR_MAP
        assert PATH_GENERATOR_MAP[MeshPath.ALTERNATING_SNAKE] is AlternatingSnakePathGenerator

    def test_path_generator_map_produces_correct_instance(self):
        from cartographer.interfaces.configuration import MeshPath
        from cartographer.macros.bed_mesh.scan_mesh import PATH_GENERATOR_MAP

        gen = PATH_GENERATOR_MAP[MeshPath.ALTERNATING_SNAKE]("x", None)
        assert isinstance(gen, AlternatingSnakePathGenerator)
