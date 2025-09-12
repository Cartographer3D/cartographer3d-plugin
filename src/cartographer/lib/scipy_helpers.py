# pyright: reportExplicitAny=false, reportUnknownVariableType=false
from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from scipy.interpolate import RBFInterpolator


def is_available() -> bool:
    """Return True if scipy is available."""
    return find_spec("scipy") is not None


def curve_fit(
    f: Callable[..., float],
    xdata: NDArray[np.float_] | list[float],
    ydata: NDArray[np.float_] | list[float],
    *,
    bounds: tuple[Any, Any] = (-np.inf, np.inf),
    maxfev: int = 10000,
    ftol: float = 1e-8,
    xtol: float = 1e-8,
) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
    """Wrapper for scipy.optimize.curve_fit, raises if unavailable."""
    if not is_available():
        msg = "scipy is required for curve fit, but is not installed."
        raise RuntimeError(msg)
    from scipy.optimize import curve_fit

    return curve_fit(f, xdata, ydata, bounds=bounds, maxfev=maxfev, ftol=ftol, xtol=xtol)


def rbf_interpolator(y: NDArray[Any], d: NDArray[Any], *, neighbors: int, smoothing: float) -> RBFInterpolator:
    if not is_available():
        msg = "scipy is required for RBF interpolation, but is not installed."
        raise RuntimeError(msg)
    from scipy.interpolate import RBFInterpolator

    return RBFInterpolator(y, d, neighbors=neighbors, smoothing=smoothing)
