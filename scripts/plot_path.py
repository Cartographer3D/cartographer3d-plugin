# pyright: reportUnknownMemberType=false
# pyright: reportUnusedCallResult=false
# pyright: reportMissingParameterType=false
# pyright: reportUnknownParameterType=false
from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from cartographer.macros.bed_mesh.paths.alternating_snake import AlternatingSnakePathGenerator
from cartographer.macros.bed_mesh.paths.hilbert_path import HilbertPathGenerator
from cartographer.macros.bed_mesh.paths.random_path import RandomPathGenerator
from cartographer.macros.bed_mesh.paths.snake_path import SnakePathGenerator
from cartographer.macros.bed_mesh.paths.spiral_path import SpiralPathGenerator

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from cartographer.macros.bed_mesh.interfaces import PathGenerator, Point

PATH_STRATEGY_MAP = {
    "snake": SnakePathGenerator,
    "alternating_snake": AlternatingSnakePathGenerator,
    "spiral": SpiralPathGenerator,
    "random": RandomPathGenerator,
    "hilbert": HilbertPathGenerator,
}

# ---------------------------------------------------------------------------
# Grid / axis helpers
# ---------------------------------------------------------------------------


def make_grid(nx: int, ny: int, spacing: float) -> list[Point]:
    """Return a flat list of (x, y) grid points ordered row by row."""
    return [(1.0 + x * spacing, 1.0 + y * spacing) for y in range(ny) for x in range(nx)]


def make_axis_limits(
    grid: list[Point],
    padding: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return x- and y-axis limits with *padding* beyond the grid extents."""
    xs = [float(p[0]) for p in grid]
    ys = [float(p[1]) for p in grid]
    return (min(xs) - padding, max(xs) + padding), (min(ys) - padding, max(ys) + padding)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def draw_grid_and_path(ax: Axes, title: str, grid: list[Point], path: list[Point]) -> None:
    """Draw grid points, path line, direction arrows, and start/end markers onto *ax*."""
    if not path:
        ax.set_title(title)
        return

    # Grid points
    gx = [float(p[0]) for p in grid]
    gy = [float(p[1]) for p in grid]
    ax.scatter(gx, gy, color="lightgray", edgecolors="gray", s=20, zorder=2, label="Grid")

    # Path line
    px = [float(p[0]) for p in path]
    py = [float(p[1]) for p in path]
    ax.plot(px, py, linestyle="-", color="steelblue", lw=1.2, zorder=1)

    # Direction arrows spaced evenly along the path
    if len(path) >= 2:
        n_arrows = min(10, len(path) - 1)
        step = max(1, (len(path) - 1) // n_arrows)
        for i in range(0, len(path) - 1, step):
            ax.annotate(
                "",
                xy=(float(path[i + 1][0]), float(path[i + 1][1])),
                xytext=(float(path[i][0]), float(path[i][1])),
                arrowprops={"arrowstyle": "->", "color": "steelblue", "lw": 1.0},
                zorder=3,
            )

    # Start / end markers
    ax.plot(float(path[0][0]), float(path[0][1]), "go", ms=6, zorder=4, label="Start")
    ax.plot(float(path[-1][0]), float(path[-1][1]), "rs", ms=6, zorder=4, label="End")

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


def render_single(
    generator: PathGenerator,
    nx: int,
    ny: int,
    spacing: float,
    axis_padding: float,
    strategy_name: str,
) -> Figure:
    """Render a single grid and return the figure."""
    fig, ax = plt.subplots(figsize=(8, 8))
    grid = make_grid(nx, ny, spacing)
    x_lim, y_lim = make_axis_limits(grid, axis_padding)
    path = list(generator.generate_path(grid, x_lim, y_lim))
    draw_grid_and_path(ax, f"{strategy_name}  {nx}×{ny}", grid, path)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_grid(value: str) -> tuple[int, int]:
    """Argparse type converter for 'NxM' grid specs (e.g. ``5x5``)."""
    parts = value.lower().split("x")
    if len(parts) != 2:
        msg = f"grid must be NxM (e.g. 5x5), got {value!r}"
        raise argparse.ArgumentTypeError(msg)
    try:
        nx, ny = int(parts[0]), int(parts[1])
    except ValueError as exc:
        msg = f"grid dimensions must be integers, got {value!r}"
        raise argparse.ArgumentTypeError(msg) from exc
    if nx < 1 or ny < 1:
        msg = f"grid dimensions must be >= 1, got {value!r}"
        raise argparse.ArgumentTypeError(msg)
    return nx, ny


def parse_max_corner_radius(value: str) -> float | None:
    """Argparse type converter: ``'auto'`` → ``None``, numeric string → non-negative float."""
    if value.lower() == "auto":
        return None
    try:
        f = float(value)
    except ValueError as exc:
        msg = f"--max-corner-radius must be 'auto' or a non-negative float, got {value!r}"
        raise argparse.ArgumentTypeError(msg) from exc
    if f < 0:
        msg = f"--max-corner-radius must be non-negative, got {f}"
        raise argparse.ArgumentTypeError(msg)
    return f


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plot_path.py",
        description="Visualise Cartographer3D bed mesh path strategies.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python scripts/plot_path.py snake
  python scripts/plot_path.py snake --grid 7x7 --max-corner-radius 0 --output snake.png
  python scripts/plot_path.py spiral --grid 9x8 --max-corner-radius auto
  python scripts/plot_path.py hilbert --grid 21x21 --direction x --output hilbert.png
""",
    )
    parser.add_argument(
        "strategy",
        choices=list(PATH_STRATEGY_MAP),
        help="Path generation strategy.",
    )
    parser.add_argument(
        "--direction",
        choices=["x", "y"],
        default="x",
        metavar="{x,y}",
        help="Primary scan direction (ignored by spiral and random). Default: x",
    )
    parser.add_argument(
        "--max-corner-radius",
        metavar="VALUE",
        default=None,
        type=parse_max_corner_radius,
        dest="max_corner_radius",
        help="Corner-radius cap: 'auto' or a non-negative float. Omit for auto (ignored by random and hilbert).",
    )
    parser.add_argument(
        "--grid",
        metavar="NXxNY",
        default="5x5",
        type=parse_grid,
        help="Grid size. Default: 5x5",
    )
    parser.add_argument(
        "--spacing",
        metavar="FLOAT",
        default=5.0,
        type=float,
        help="Point-to-point spacing. Default: 5",
    )
    parser.add_argument(
        "--axis-padding",
        metavar="FLOAT",
        default=2.0,
        dest="axis_padding",
        type=float,
        help="Axis padding beyond grid extents. Default: 2",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        help="Save static image (PNG, PDF, SVG, ...).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Force interactive display even when --output is given.",
    )
    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    strategy_cls = PATH_STRATEGY_MAP[args.strategy]
    generator: PathGenerator = strategy_cls(args.direction, args.max_corner_radius)

    nx, ny = args.grid  # tuple[int, int] from parse_grid

    fig = render_single(generator, nx, ny, args.spacing, args.axis_padding, args.strategy)

    if args.output is not None:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.output!r}")

    if args.show or args.output is None:
        plt.show()


if __name__ == "__main__":
    main()
