from __future__ import annotations


def param_linear(x: float, a: float, b: float) -> float:
    return a * x + b


def line_fit(x: float, a: float, b: float, c: float) -> float:
    """Quadratic fit function."""
    return a * x**2 + b * x + c


def line0(x: float, a: float, c: float) -> float:
    """Quadratic fit with b=0."""
    return a * x**2 + c


def line120(x: float, a: float, c: float) -> float:
    """Quadratic fit with vertex at x=120."""
    return a * x**2 - 240 * a * x + c
