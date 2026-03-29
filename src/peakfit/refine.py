"""Recursive window refinement: polynomial fit → extremum → tighter window → repeat."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from peakfit.continuum import ContinuumMethod, continuum_remove
from peakfit.polynomial import Mode, extremum_from_fit, fit_polynomial


@dataclass(frozen=True)
class PeakRefinementResult:
    """Result of subsample peak refinement."""

    peak_x: float
    coeffs: np.ndarray
    iterations: int
    converged: bool
    peak_x_history: tuple[float, ...]


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
    peak_x = extremum_from_fit(coeffs, degree, mode)
    history.append(peak_x)

    converged = False
    it_done = 1
    for _ in range(1, n_iterations):
        idx = _nearest_index(xw, peak_x)
        xw, yw = subset_around_index(xw, yw, idx, half_width)
        if xw.size < min_points:
            break
        prev = peak_x
        coeffs = fit_polynomial(xw, yw, degree)
        peak_x = extremum_from_fit(coeffs, degree, mode)
        history.append(peak_x)
        it_done += 1
        if abs(peak_x - prev) < atol:
            converged = True
            break

    return PeakRefinementResult(
        peak_x=float(peak_x),
        coeffs=coeffs,
        iterations=it_done,
        converged=converged,
        peak_x_history=tuple(history),
    )
