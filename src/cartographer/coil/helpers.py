from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


def param_linear(x: float, a: float, b: float) -> float:
    return a * x + b


def linear_func(x: NDArray[np.float_], a: float, b: float) -> NDArray[np.float_]:
    return a * x + b


def line_fit(x: NDArray[np.float_], a: float, b: float, c: float) -> NDArray[np.float_]:
    """Quadratic fit function."""
    return a * x**2 + b * x + c


def line0(x: NDArray[np.float_], a: float, c: float) -> NDArray[np.float_]:
    """Quadratic fit with b=0."""
    return a * x**2 + c


def line120(x: NDArray[np.float_], a: float, c: float) -> NDArray[np.float_]:
    """Quadratic fit with vertex at x=120."""
    return a * x**2 - 240 * a * x + c
