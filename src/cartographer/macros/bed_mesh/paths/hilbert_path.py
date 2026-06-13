from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, final

from typing_extensions import override

from cartographer.macros.bed_mesh.interfaces import PathGenerator
from cartographer.macros.bed_mesh.paths.utils import cluster_points

if TYPE_CHECKING:
    from cartographer.macros.bed_mesh.interfaces import Point


def next_power_of_two(n: int) -> int:
    """Return the smallest power of two that is >= *n*."""
    if n <= 1:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p


def xy_to_hilbert(order: int, x: int, y: int) -> int:
    """Convert (x, y) grid coordinates to a Hilbert curve distance.

    ``order`` is the side length of the embedding square (must be a power of
    two).  Both ``x`` and ``y`` must satisfy ``0 <= x, y < order``.

    The algorithm follows the standard d2xy / xy2d mapping described at
    https://en.wikipedia.org/wiki/Hilbert_curve#Applications_and_mapping_algorithms
    """
    d = 0
    s = order >> 1
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        # Rotate the quadrant so the next level is in the right orientation.
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        s >>= 1
    return d


@final
class HilbertPathGenerator(PathGenerator):
    """Scan the bed mesh in Hilbert-curve order.

    Grid points are assigned integer (col, row) coordinates, embedded inside
    the smallest power-of-two square that contains the full grid, and then
    sorted by their Hilbert index in that square.  This minimises the maximum
    step between consecutive scan points while still covering every grid point
    exactly once — no duplicate waypoints, no arc interpolation.

    ``main_direction`` controls the orientation of the embedding:

    * ``"x"`` — the column index is mapped to the *y* Hilbert axis and the row
      index is mapped to the *x* Hilbert axis so the very first segment of the
      curve moves in the +X physical direction.
    * ``"y"`` — transposed: the first segment moves in the +Y physical
      direction.

    ``max_corner_radius`` is accepted for API compatibility with other
    generators but has no effect — the Hilbert path visits only the original
    mesh points without inserting arc waypoints.
    """

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
        del x_axis_limits, y_axis_limits  # not used — no arc waypoints

        if not points:
            return

        # Build a 2-D grid: grid[r][c] is the physical point at y-row r, x-col c.
        grid = cluster_points(points, "x")
        rows = len(grid)
        cols = max(len(row) for row in grid) if rows else 0

        order = next_power_of_two(max(cols, rows))

        keyed: list[tuple[int, Point]] = []
        for r, row in enumerate(grid):
            for c, pt in enumerate(row):
                # main_direction="x": (row→x_hilbert, col→y_hilbert) so first step is +physical-X.
                # main_direction="y": (col→x_hilbert, row→y_hilbert) so first step is +physical-Y.
                key = xy_to_hilbert(order, r, c) if self.main_direction == "x" else xy_to_hilbert(order, c, r)
                keyed.append((key, pt))

        keyed.sort(key=lambda item: item[0])
        for _, pt in keyed:
            yield pt
