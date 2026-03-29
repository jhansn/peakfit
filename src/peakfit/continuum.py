"""Optional continuum removal (spectral normalization before peak fitting)."""

from __future__ import annotations

from typing import Literal

import numpy as np

ContinuumMethod = Literal["none", "linear", "hull"]


def _cross2(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def _upper_hull_indices(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Indices of vertices of the upper convex hull (x assumed sorted ascending)."""
    n = x.size
    if n <= 2:
        return np.arange(n, dtype=np.intp)
    pts = np.column_stack([x, y])
    upper: list[int] = [0, 1]
    for i in range(2, n):
        while len(upper) >= 2:
            o, a = upper[-2], upper[-1]
            if _cross2(pts[o], pts[a], pts[i]) >= 0.0:
                upper.pop()
            else:
                break
        upper.append(i)
    return np.array(upper, dtype=np.intp)


def continuum_remove(
    x: np.ndarray,
    y: np.ndarray,
    method: Literal["linear", "hull"] = "linear",
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Divide spectra by an estimated continuum (continuum removal / hull quotient).

    Returns ``y / continuum(x)`` with the same shape as ``y``. Values are
    typically near 1 on the continuum and below 1 in absorption troughs
    (reflectance convention).

    Parameters
    ----------
    x
        Strictly increasing coordinates (e.g. wavelength or band index).
    y
        Same-length spectral values (e.g. reflectance).
    method
        ``\"linear\"``: straight-line continuum between first and last point.
        ``\"hull\"``: piecewise-linear upper convex hull (standard hull quotient).
    eps
        Floor for continuum to avoid division by zero.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    if x.size < 2:
        raise ValueError("need at least 2 points for continuum removal")
    if np.any(np.diff(x) <= 0):
        raise ValueError("x must be strictly increasing")

    if method == "linear":
        x0, x1 = x[0], x[-1]
        y0, y1 = y[0], y[-1]
        cont = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    elif method == "hull":
        idx = _upper_hull_indices(x, y)
        cont = np.interp(x, x[idx], y[idx])
    else:
        raise ValueError('method must be "linear" or "hull"')

    cont = np.maximum(cont, eps)
    return y / cont
