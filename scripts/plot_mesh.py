# pyright: reportUnknownMemberType=false
# pyright: reportUnusedCallResult=false
# pyright: reportMissingParameterType=false
# pyright: reportUnknownParameterType=false
from __future__ import annotations

import configparser
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

from cartographer.interfaces.printer import Position
from cartographer.macros.bed_mesh.helpers import CoordinateTransformer, Region

# Type aliases for clarity
FaultyRegionList = List[Region]
PositionList = List[Position]


def load_bed_mesh(
    path: str, section: str = "bed_mesh"
) -> tuple[PositionList, int, int, FaultyRegionList, float, float, float, float]:
    """Load bed mesh data and faulty regions from a config file.

    Returns:
        positions: List[Position]
        x_count: int
        y_count: int
        faulty_regions: List[Region]
        min_x, max_x, min_y, max_y: float
    """
    config = configparser.ConfigParser()
    config.read(path)
    sec = config[section]

    x_count = int(sec["x_count"])
    y_count = int(sec["y_count"])
    points_raw = sec["points"].strip().splitlines()
    points = [[float(v) for v in line.strip().split(",")] for line in points_raw]

    if len(points) != y_count or any(len(row) != x_count for row in points):
        msg = "points does not match x_count/y_count"
        raise ValueError(msg)

    min_x = float(sec.get("min_x", 0.0))
    max_x = float(sec.get("max_x", x_count - 1))
    min_y = float(sec.get("min_y", 0.0))
    max_y = float(sec.get("max_y", y_count - 1))

    xs = np.linspace(min_x, max_x, x_count)
    ys = np.linspace(min_y, max_y, y_count)

    faulty_regions: FaultyRegionList = []
    for i in range(1, 100):
        min_key = f"faulty_region_{i}_min"
        max_key = f"faulty_region_{i}_max"
        if min_key not in sec or max_key not in sec:
            continue
        min_vals = [float(v) for v in sec[min_key].split(",")]
        max_vals = [float(v) for v in sec[max_key].split(",")]
        faulty_regions.append(Region((min_vals[0], min_vals[1]), (max_vals[0], max_vals[1])))

    positions: PositionList = [Position(x, y, points[j][i]) for j, y in enumerate(ys) for i, x in enumerate(xs)]

    return positions, x_count, y_count, faulty_regions, min_x, max_x, min_y, max_y


transformer = CoordinateTransformer(Position(0, 0, 0))


def visualize_faulty_interpolation(
    positions: PositionList,
    faulty_regions: FaultyRegionList,
    size_x: int,
    size_y: int,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
) -> None:
    """Visualize the mesh and interpolation of faulty regions."""
    Z = np.asarray([p.z for p in positions], dtype=float).reshape(size_y, size_x)  # noqa: N806  # pyright: ignore[reportUnknownVariableType]

    output = transformer.apply_faulty_regions(positions, faulty_regions=faulty_regions)
    Z_interp = np.asarray([p.z for p in output], dtype=float).reshape(size_y, size_x)  # noqa: N806  # pyright: ignore[reportUnknownVariableType]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("Bed Mesh Faulty Region Interpolation")

    # Raw mesh
    ax = axes[0]
    im = ax.imshow(
        Z,
        origin="lower",
        cmap="viridis",
        extent=[min_x, max_x, min_y, max_y],
        aspect="auto",
    )
    ax.set_title("Raw mesh (with faulty regions)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for region in faulty_regions:
        ax.add_patch(
            patches.Rectangle(
                (float(region.min_point[0]), float(region.min_point[1])),
                float(region.max_point[0] - region.min_point[0]),
                float(region.max_point[1] - region.min_point[1]),
                fill=False,
                edgecolor="red",
                lw=2,
            )
        )

    # Interpolated mesh
    ax = axes[1]
    im2 = ax.imshow(
        Z_interp,
        origin="lower",
        cmap="viridis",
        extent=[min_x, max_x, min_y, max_y],
        aspect="auto",
    )
    ax.set_title("Interpolated mesh")
    plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

    plt.show()


if __name__ == "__main__":
    positions, size_x, size_y, faulty_regions, min_x, max_x, min_y, max_y = load_bed_mesh("./scripts/bed_mesh.conf")
    visualize_faulty_interpolation(positions, faulty_regions, size_x, size_y, min_x, max_x, min_y, max_y)
