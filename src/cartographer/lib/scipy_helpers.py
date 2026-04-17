# pyright: reportExplicitAny=false, reportUnknownVariableType=false
from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from scipy.interpolate import RBFInterpolator


def _raise_if_scipy_unavailable(module: str, name: str) -> None:
    """Private helper to check scipy module/name availability."""
    if find_spec("scipy") is None:
        msg = "scipy is required but is not installed."
        raise RuntimeError(msg)
    try:
        mod = import_module(module)
        getattr(mod, name)
    except (ImportError, AttributeError) as e:
        msg = (
            "scipy is installed but failed to import "
            "(possibly due to incompatible numpy version or broken install). "
            f"{e!s}\nTry: pip install --upgrade scipy"
        )
        raise RuntimeError(msg) from None


def raise_if_curve_fit_unavailable() -> None:
    """Raise with clear error if scipy.optimize.curve_fit is unavailable."""
    _raise_if_scipy_unavailable("scipy.optimize", "curve_fit")


def raise_if_rbf_interpolator_unavailable() -> None:
    """Raise with clear error if scipy.interpolate.RBFInterpolator is unavailable."""
    _raise_if_scipy_unavailable("scipy.interpolate", "RBFInterpolator")


def is_available() -> bool:
    """Return True if scipy is available."""
    try:
        raise_if_curve_fit_unavailable()
        return True
    except RuntimeError:
        return False


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
    raise_if_curve_fit_unavailable()
    try:
        from scipy.optimize import curve_fit as scipy_curve_fit
    except (ImportError, AttributeError) as e:
        msg = (
            "scipy.optimize is installed but failed to import "
            "(possibly due to incompatible numpy version or broken install). "
            f"{e!s}\nTry: pip install --upgrade scipy"
        )
        raise RuntimeError(msg) from None

    return scipy_curve_fit(f, xdata, ydata, bounds=bounds, maxfev=maxfev, ftol=ftol, xtol=xtol)


def rbf_interpolator(y: NDArray[Any], d: NDArray[Any], *, neighbors: int, smoothing: float) -> RBFInterpolator:
    """Wrapper for scipy.interpolate.RBFInterpolator, raises if unavailable."""
    raise_if_rbf_interpolator_unavailable()
    try:
        from scipy.interpolate import RBFInterpolator
    except (ImportError, AttributeError) as e:
        msg = (
            "scipy.interpolate is installed but failed to import "
            "(possibly due to incompatible numpy version or broken install). "
            f"{e!s}\nTry: pip install --upgrade scipy"
        )
        raise RuntimeError(msg) from None

    return RBFInterpolator(y, d, neighbors=neighbors, smoothing=smoothing)
