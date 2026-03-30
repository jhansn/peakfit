"""Optional continuum removal (spectral normalization before peak fitting)."""

from __future__ import annotations

from typing import Literal

import numpy as np

ContinuumMethod = Literal["none", "linear", "hull"]

try:
    from numba import njit, prange
except Exception:  # pragma: no cover - optional acceleration dependency
    njit = None
    prange = range


def _cross2(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def _upper_hull_indices_py(x: np.ndarray, y: np.ndarray) -> np.ndarray:
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


if njit is not None:
    @njit(cache=True)
    def _upper_hull_indices_jit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """JIT-compiled upper-hull index extraction using an array-backed stack."""
        n = x.size
        if n <= 2:
            out = np.empty(n, dtype=np.int64)
            for i in range(n):
                out[i] = i
            return out

        upper = np.empty(n, dtype=np.int64)
        m = 0
        upper[m] = 0
        m += 1
        upper[m] = 1
        m += 1

        for i in range(2, n):
            while m >= 2:
                o = upper[m - 2]
                a = upper[m - 1]
                cross = (x[a] - x[o]) * (y[i] - y[o]) - (y[a] - y[o]) * (x[i] - x[o])
                if cross >= 0.0:
                    m -= 1
                else:
                    break
            upper[m] = i
            m += 1
        return upper[:m]
else:
    _upper_hull_indices_jit = None


def _upper_hull_indices(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Dispatch to JIT hull extraction when Numba is available."""
    if _upper_hull_indices_jit is not None:
        return _upper_hull_indices_jit(x, y)
    return _upper_hull_indices_py(x, y)


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


if njit is not None:
    @njit(cache=True, parallel=True)
    def _continuum_remove_hull_rows_jit(
        x: np.ndarray,
        y2d: np.ndarray,
        eps: float,
    ) -> np.ndarray:
        """JIT-compiled hull continuum removal for many spectra (rows)."""
        n_rows, n = y2d.shape
        out = np.empty((n_rows, n), dtype=np.float64)
        for r in prange(n_rows):
            y = y2d[r]

            # Upper hull indices via monotone chain (array-backed stack).
            upper = np.empty(n, dtype=np.int64)
            m = 0
            upper[m] = 0
            m += 1
            upper[m] = 1
            m += 1
            for i in range(2, n):
                while m >= 2:
                    o = upper[m - 2]
                    a = upper[m - 1]
                    cross = (x[a] - x[o]) * (y[i] - y[o]) - (y[a] - y[o]) * (x[i] - x[o])
                    if cross >= 0.0:
                        m -= 1
                    else:
                        break
                upper[m] = i
                m += 1

            # Piecewise-linear interpolation on hull segments + safe divide.
            seg = 0
            for i in range(n):
                while (seg + 1) < (m - 1) and i > upper[seg + 1]:
                    seg += 1
                i0 = upper[seg]
                i1 = upper[seg + 1]
                if i1 == i0:
                    cont = y[i0]
                else:
                    t = (x[i] - x[i0]) / (x[i1] - x[i0])
                    cont = y[i0] + (y[i1] - y[i0]) * t
                if cont < eps:
                    cont = eps
                out[r, i] = y[i] / cont
        return out
else:
    _continuum_remove_hull_rows_jit = None


def continuum_remove_rows(
    x: np.ndarray,
    y2d: np.ndarray,
    method: Literal["linear", "hull"] = "linear",
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Apply continuum removal to a batch of spectra (shape ``(n_rows, n)``)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y2d = np.asarray(y2d, dtype=np.float64)
    if y2d.ndim != 2:
        raise ValueError("y2d must have shape (n_rows, n)")
    if y2d.shape[1] != x.size:
        raise ValueError("x length must match y2d.shape[1]")
    if x.size < 2:
        raise ValueError("need at least 2 points for continuum removal")
    if np.any(np.diff(x) <= 0):
        raise ValueError("x must be strictly increasing")

    if method == "linear":
        x0, x1 = x[0], x[-1]
        t = ((x - x0) / (x1 - x0)).reshape(1, -1)
        y0 = y2d[:, [0]]
        y1 = y2d[:, [-1]]
        cont = y0 + (y1 - y0) * t
        cont = np.maximum(cont, eps)
        return y2d / cont

    if method == "hull":
        if _continuum_remove_hull_rows_jit is not None:
            return _continuum_remove_hull_rows_jit(x, y2d, eps)
        out = np.empty_like(y2d, dtype=np.float64)
        for i in range(y2d.shape[0]):
            out[i] = continuum_remove(x, y2d[i], method="hull", eps=eps)
        return out

    raise ValueError('method must be "linear" or "hull"')
