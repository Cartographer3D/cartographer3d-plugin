from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING, Iterator, Literal, cast

import numpy as np
from typing_extensions import TypeAlias

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from cartographer.macros.bed_mesh.interfaces import Point

Vec: TypeAlias = "NDArray[np.float_]"


def cluster_points(points: list[Point], axis: Literal["x", "y"], tol: float = 1e-3) -> list[list[Point]]:
    # axis to cluster on:
    # if main_direction = "x", cluster on y (index 1)
    # if main_direction = "y", cluster on x (index 0)
    cluster_index = 1 if axis == "x" else 0
    sort_index = 0 if axis == "x" else 1

    clusters: dict[float, list[Point]] = defaultdict(list)
    for p in points:
        key = round(p[cluster_index] / tol)
        clusters[key].append(p)

    sorted_keys = sorted(clusters.keys())

    rows: list[list[Point]] = []
    for key in sorted_keys:
        row_points = clusters[key]
        row_points.sort(key=lambda pt: pt[sort_index])
        rows.append(row_points)

    return rows


def arc_points(
    center: Vec, radius: float, start_angle_deg: float, span_deg: float, max_dev: float = 0.1
) -> Iterator[Point]:
    cx, cy = center
    if radius == 0:
        yield (cx, cy)
        return

    max_dev = min(max_dev, radius)  # Avoid domain error in arccos
    start_rad = np.deg2rad(start_angle_deg)
    span_rad = np.deg2rad(span_deg)

    d_theta = np.arccos(1 - max_dev / radius)
    n_points = max(1, int(np.ceil(abs(span_rad) / d_theta)))
    thetas = cast("NDArray[np.float_]", start_rad + np.linspace(0, span_rad, n_points + 1))

    xs = cx + radius * np.cos(thetas)
    ys = cy + radius * np.sin(thetas)

    yield from zip(xs, ys)


def perpendicular(v: Vec, ccw: bool = True) -> Vec:
    return np.asarray([-v[1], v[0]]) if ccw else np.asarray([v[1], -v[0]])


def angle_deg(v: Vec) -> float:
    return math.degrees(math.atan2(v[1], v[0]))


def normalize(v: Vec) -> Vec:
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v


def row_direction(row: list[Point]) -> Vec:
    if len(row) < 2:
        msg = "Need at least two points to determine direction"
        raise ValueError(msg)
    p0: Vec = np.asarray(row[0], dtype=float)
    p1: Vec = np.asarray(row[1], dtype=float)
    dir_vec = p1 - p0
    return dir_vec / np.linalg.norm(dir_vec)  # normalized
