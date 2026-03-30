"""Recursive window refinement: polynomial fit → extremum → tighter window → repeat."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from peakfit.continuum import ContinuumMethod, continuum_remove
from peakfit.polynomial import ExtremumPoint, Mode, extrema_from_fit, fit_polynomial


@dataclass(frozen=True)
class PeakRefinementResult:
    """Result of subsample peak refinement."""

    peak_x: float
    coeffs: np.ndarray
    extrema: tuple[ExtremumPoint, ...]
    iterations: int
    converged: bool
    peak_x_history: tuple[float, ...]


def _select_extremum(extrema: tuple[ExtremumPoint, ...], mode: Mode, degree: int) -> float:
    """Select requested extremum type from precomputed extrema."""
    chosen = [e for e in extrema if e.kind == mode]
    if chosen:
        return chosen[0].x
    if degree == 2:
        if mode == "max":
            raise ValueError("quadratic opens upward: no local maximum")
        raise ValueError("quadratic opens downward: no local minimum")
    if mode == "max":
        raise ValueError("no real local maximum on cubic")
    raise ValueError("no real local minimum on cubic")


def _nearest_index(x: np.ndarray, x0: float) -> int:
    return int(np.argmin(np.abs(x - x0)))


def subset_around_index(
    x: np.ndarray,
    y: np.ndarray,
    center_index: int,
    half_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep ``2*half_width + 1`` points centered on ``center_index`` (clipped)."""
    n = x.size
    lo = max(0, center_index - half_width)
    hi = min(n, center_index + half_width + 1)
    return x[lo:hi].copy(), y[lo:hi].copy()


def refine_peak_subsample(
    x: np.ndarray,
    y: np.ndarray,
    *,
    degree: Literal[2, 3] = 2,
    n_iterations: int = 3,
    half_width: int = 3,
    adaptive_window: bool = False,
    half_width_min: int = 3,
    half_width_shrink: float = 0.65,
    mode: Mode = "max",
    min_points: int | None = None,
    atol: float = 1e-6,
    continuum: ContinuumMethod = "none",
    continuum_eps: float = 1e-12,
) -> PeakRefinementResult:
    """Fit polynomial, find extremum, subset around it, refit (recursive windowing).

    ``x`` may be band indices or wavelengths; output ``peak_x`` uses the same
    units. Use :func:`peakfit.wavelength.index_to_wavelength` to convert
    fractional indices to nm.

    Parameters
    ----------
    x, y
        Same length; ``y`` is spectral response in the window.
    degree
        2 (quadratic / parabolic) or 3 (cubic).
    n_iterations
        Number of fit–subset cycles.
    half_width
        After each fit, keep the ``2*half_width+1`` samples closest to the
        fitted extremum (before clipping to array bounds).
    adaptive_window
        If True, shrink ``half_width`` each iteration by ``half_width_shrink``
        down to ``half_width_min`` to improve locality near convergence.
    half_width_min
        Minimum half-width used when ``adaptive_window=True``.
    half_width_shrink
        Multiplicative shrink factor applied per iteration when
        ``adaptive_window=True``.
    mode
        Track a local ``"max"`` or ``"min"``.
    min_points
        Minimum samples to keep; defaults to ``degree + 2``.
    atol
        Convergence: stop early if successive ``peak_x`` differ by less than
        ``atol`` (in ``x`` units).
    continuum
        If not ``\"none\"``, divide ``y`` by a continuum estimate once before
        fitting: ``\"linear\"`` (endpoints line) or ``\"hull\"`` (upper convex
        hull). Same units as input; peak positions refer to the
        continuum-removed curve.
    continuum_eps
        Denominator floor passed to :func:`peakfit.continuum.continuum_remove`.
    """
    if min_points is None:
        min_points = degree + 2
    if half_width < 1:
        raise ValueError("half_width must be >= 1")
    if half_width_min < 1:
        raise ValueError("half_width_min must be >= 1")
    if not (0.0 < half_width_shrink <= 1.0):
        raise ValueError("half_width_shrink must be in (0, 1]")

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    if x.size < min_points:
        raise ValueError(f"need at least {min_points} points")

    if continuum != "none":
        y = continuum_remove(x, y, method=continuum, eps=continuum_eps)

    xw, yw = x.copy(), y.copy()
    history: list[float] = []
    coeffs = fit_polynomial(xw, yw, degree)
    extrema = extrema_from_fit(coeffs, degree)
    peak_x = _select_extremum(extrema, mode, degree)
    history.append(peak_x)

    converged = False
    it_done = 1
    for it in range(1, n_iterations):
        if adaptive_window:
            hw = max(
                half_width_min,
                int(round(half_width * (half_width_shrink ** (it - 1)))),
            )
        else:
            hw = half_width
        idx = _nearest_index(xw, peak_x)
        prev_lo, prev_hi = float(xw[0]), float(xw[-1])
        xw_next, yw_next = subset_around_index(xw, yw, idx, hw)
        new_lo, new_hi = float(xw_next[0]), float(xw_next[-1])
        xw, yw = xw_next, yw_next
        if xw.size < min_points:
            break
        # If bounds no longer change, next fit is identical and root won't move.
        if new_lo == prev_lo and new_hi == prev_hi:
            converged = True
            break
        prev = peak_x
        coeffs = fit_polynomial(xw, yw, degree)
        extrema = extrema_from_fit(coeffs, degree)
        peak_x = _select_extremum(extrema, mode, degree)
        history.append(peak_x)
        it_done += 1
        if abs(peak_x - prev) < atol:
            converged = True
            break

    return PeakRefinementResult(
        peak_x=float(peak_x),
        coeffs=coeffs,
        extrema=extrema,
        iterations=it_done,
        converged=converged,
        peak_x_history=tuple(history),
    )


def fit_extrema_subsample(
    x: np.ndarray,
    y: np.ndarray,
    *,
    degree: Literal[2, 3] = 2,
    continuum: ContinuumMethod = "none",
    continuum_eps: float = 1e-12,
) -> tuple[ExtremumPoint, ...]:
    """Fit one local polynomial and return all extrema with type labels.

    This is a mode-free helper for bulk workflows:
    fit once, then filter returned extrema by ``kind`` (``"min"``/``"max"``).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    if x.size < degree + 2:
        raise ValueError(f"need at least {degree + 2} points")
    if continuum != "none":
        y = continuum_remove(x, y, method=continuum, eps=continuum_eps)
    coeffs = fit_polynomial(x, y, degree)
    return extrema_from_fit(coeffs, degree)
