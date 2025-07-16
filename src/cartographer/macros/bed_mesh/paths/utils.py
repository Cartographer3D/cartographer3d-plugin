from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Iterator, Literal, cast

import numpy as np
from typing_extensions import TypeAlias

if TYPE_CHECKING:
    from cartographer.macros.bed_mesh.interfaces import Point

Vec: TypeAlias = "np.ndarray[Literal[2], np.dtype[np.float64]]"


def arc_points(
    center: Vec, radius: float, start_angle_deg: float, span_deg: float, max_dev: float = 0.1
) -> Iterator[Point]:
    if radius == 0:
        return

    max_dev = min(max_dev, radius)  # Avoid domain error in arccos
    start_rad = np.deg2rad(start_angle_deg)
    span_rad = np.deg2rad(span_deg)

    d_theta = np.arccos(1 - max_dev / radius)
    n_points = max(1, int(np.ceil(abs(span_rad) / d_theta)))
    thetas = cast("np.ndarray[Any, np.dtype[np.float64]]", start_rad + np.linspace(0, span_rad, n_points + 1))  # pyright: ignore[reportExplicitAny]

    cx, cy = center
    xs = cx + radius * np.cos(thetas)
    ys = cy + radius * np.sin(thetas)

    yield from zip(xs, ys)


def perpendicular(v: Vec, ccw: bool = True) -> Vec:
    return np.array([-v[1], v[0]]) if ccw else np.array([v[1], -v[0]])


def angle_deg(v: Vec) -> float:
    return math.degrees(math.atan2(v[1], v[0]))


def normalize(v: Vec) -> Vec:
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v


def row_direction(row: list[Point]) -> Vec:
    if len(row) < 2:
        msg = "Need at least two points to determine direction"
        raise ValueError(msg)
    p0: Vec = np.array(row[0], dtype=float)
    p1: Vec = np.array(row[1], dtype=float)
    dir_vec = p1 - p0
    return dir_vec / np.linalg.norm(dir_vec)  # normalized
