"""Local polynomial fits and analytic extrema (quadratic and cubic)."""

from __future__ import annotations

from typing import Literal

import numpy as np

Mode = Literal["max", "min"]

# Coefficient order from ``numpy.polyfit``: highest degree first.


def fit_polynomial(
    x: np.ndarray,
    y: np.ndarray,
    degree: int,
    *,
    rcond: float | None = None,
) -> np.ndarray:
    """Least-squares polynomial; returns coeffs high-to-low (``polyfit`` order)."""
    if degree not in (2, 3):
        raise ValueError("degree must be 2 (quadratic) or 3 (cubic)")
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    n = x.size
    if n <= degree:
        raise ValueError(f"need at least {degree + 1} points for degree {degree}")
    return np.polyfit(x, y, degree, rcond=rcond)


def quadratic_vertex(coeffs: np.ndarray) -> float:
    """Analytic vertex of ``a*x**2 + b*x + c`` with coeffs ``[a, b, c]``.

    Vertex at ``x = -b / (2*a)``. Raises if ``|a|`` is numerically zero.
    """
    a, b, _ = coeffs[0], coeffs[1], coeffs[2]
    if np.abs(a) < 1e-14:
        raise ValueError("degenerate quadratic (a ~= 0)")
    return float(-b / (2.0 * a))


def cubic_extrema(coeffs: np.ndarray) -> tuple[float, float]:
    """Real critical points of cubic; returns ``(x_max, x_min)`` among real roots.

    Coeffs ``[a,b,c,d]`` for ``a*x**3 + b*x**2 + c*x + d``. If only one
    extremum is real, the other slot contains ``nan``.
    """
    a, b, c, _ = coeffs[0], coeffs[1], coeffs[2], coeffs[3]
    if np.abs(a) < 1e-14:
        raise ValueError("degenerate cubic (leading coefficient ~= 0)")

    # First derivative: 3*a*x**2 + 2*b*x + c
    r = np.roots(np.array([3.0 * a, 2.0 * b, c], dtype=np.complex128))
    real_mask = np.abs(r.imag) < 1e-10
    xs = np.real(r[real_mask])
    if xs.size == 0:
        return float("nan"), float("nan")

    # Second derivative of cubic: 6*a*x + 2*b
    xmax, xmin = float("nan"), float("nan")
    for xv in xs:
        curv = 6.0 * a * float(xv) + 2.0 * b
        if curv < 0:
            xmax = float(xv)
        elif curv > 0:
            xmin = float(xv)
    return xmax, xmin


def extremum_from_fit(
    coeffs: np.ndarray,
    degree: int,
    mode: Mode,
) -> float:
    """Pick the local maximum or minimum of the fitted polynomial."""
    if degree == 2:
        x0 = quadratic_vertex(coeffs)
        a = coeffs[0]
        if mode == "max" and a > 0:
            raise ValueError("quadratic opens upward: no local maximum")
        if mode == "min" and a < 0:
            raise ValueError("quadratic opens downward: no local minimum")
        return x0

    if degree == 3:
        x_max, x_min = cubic_extrema(coeffs)
        if mode == "max":
            if np.isnan(x_max):
                raise ValueError("no real local maximum on cubic")
            return x_max
        if np.isnan(x_min):
            raise ValueError("no real local minimum on cubic")
        return x_min

    raise ValueError("degree must be 2 or 3")
