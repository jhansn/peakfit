"""Hyperspectral cube I/O and per-pixel peak mapping."""

from __future__ import annotations

from typing import Literal

import numpy as np

from peakfit.continuum import ContinuumMethod, continuum_remove_rows
from peakfit.polynomial import Mode
from peakfit.refine import PeakRefinementResult, refine_peak_subsample
from peakfit.wavelength import index_to_wavelength


def _window_sum(
    prefix: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    rows: np.ndarray | None = None,
) -> np.ndarray:
    """Window sums for many rows using prefix sums with per-row bounds."""
    if rows is None:
        rows = np.arange(prefix.shape[0], dtype=np.intp)
    return prefix[rows, hi] - prefix[rows, lo]


def _nearest_int_tie_low(x: np.ndarray) -> np.ndarray:
    """Nearest integer with ties resolved toward the lower integer."""
    lo = np.floor(x)
    hi = lo + 1.0
    d_lo = np.abs(x - lo)
    d_hi = np.abs(hi - x)
    return np.where(d_hi < d_lo, hi, lo).astype(np.int64)


def _peak_map_quadratic_batched(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    band_start: int,
    band_stop: int,
    *,
    n_iterations: int,
    half_width: int,
    mode: Mode,
    output_wavelength: bool,
    continuum: ContinuumMethod,
    continuum_eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Batched quadratic peak/min map with iterative window shifts."""
    m, n = cube.shape[0], cube.shape[1]
    peaks = np.full((m, n), np.nan, dtype=np.float64)
    valid = np.zeros((m, n), dtype=bool)

    x = np.arange(band_start, band_stop, dtype=np.float64)
    k = x.size
    p = m * n
    if k < 4:
        return peaks, valid

    y2d = np.asarray(cube[:, :, band_start:band_stop], dtype=np.float64).reshape(p, k)

    if continuum in ("linear", "hull"):
        y2d = continuum_remove_rows(x, y2d, method=continuum, eps=continuum_eps)

    # Prefix sums for y, x*y, x^2*y for O(1) window extraction per pixel.
    x1 = x
    x2 = x1 * x1
    x3 = x2 * x1
    x4 = x2 * x2

    py0 = np.pad(np.cumsum(y2d, axis=1), ((0, 0), (1, 0)))
    py1 = np.pad(np.cumsum(y2d * x1[None, :], axis=1), ((0, 0), (1, 0)))
    py2 = np.pad(np.cumsum(y2d * x2[None, :], axis=1), ((0, 0), (1, 0)))

    # Prefix sums for x powers (shared by all pixels).
    px1 = np.pad(np.cumsum(x1), (1, 0))
    px2 = np.pad(np.cumsum(x2), (1, 0))
    px3 = np.pad(np.cumsum(x3), (1, 0))
    px4 = np.pad(np.cumsum(x4), (1, 0))

    lo = np.zeros(p, dtype=np.int64)
    hi = np.full(p, k, dtype=np.int64)
    peak_x = np.full(p, np.nan, dtype=np.float64)

    # Initial fit on full window.
    npts = (hi - lo).astype(np.float64)
    s1 = px1[hi] - px1[lo]
    s2 = px2[hi] - px2[lo]
    s3 = px3[hi] - px3[lo]
    s4 = px4[hi] - px4[lo]
    t0 = _window_sum(py0, lo, hi)
    t1 = _window_sum(py1, lo, hi)
    t2 = _window_sum(py2, lo, hi)

    a_mat = np.empty((p, 3, 3), dtype=np.float64)
    b_mat = np.empty((p, 3, 1), dtype=np.float64)
    a_mat[:, 0, 0] = s4
    a_mat[:, 0, 1] = s3
    a_mat[:, 0, 2] = s2
    a_mat[:, 1, 0] = s3
    a_mat[:, 1, 1] = s2
    a_mat[:, 1, 2] = s1
    a_mat[:, 2, 0] = s2
    a_mat[:, 2, 1] = s1
    a_mat[:, 2, 2] = npts
    b_mat[:, 0, 0] = t2
    b_mat[:, 1, 0] = t1
    b_mat[:, 2, 0] = t0

    coeffs = np.linalg.solve(a_mat, b_mat)[:, :, 0]
    a = coeffs[:, 0]
    b = coeffs[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        peak_x = -b / (2.0 * a)

    alive = np.isfinite(peak_x) & (np.abs(a) >= 1e-14)
    if mode == "max":
        alive &= a < 0.0
    else:
        alive &= a > 0.0

    can_iter = alive.copy()
    for _ in range(1, n_iterations):
        idx_rows = np.where(can_iter)[0]
        if idx_rows.size == 0:
            break

        idx_near = _nearest_int_tie_low(peak_x[idx_rows] - float(band_start))
        lo_i = lo[idx_rows]
        hi_i = hi[idx_rows]
        idx_near = np.clip(idx_near, lo_i, hi_i - 1)

        new_lo = np.maximum(lo_i, idx_near - half_width)
        new_hi = np.minimum(hi_i, idx_near + half_width + 1)
        nnew = new_hi - new_lo

        fit_ok = nnew >= 4
        stop_rows = idx_rows[~fit_ok]
        can_iter[stop_rows] = False

        fit_rows = idx_rows[fit_ok]
        if fit_rows.size == 0:
            continue

        lo[fit_rows] = new_lo[fit_ok]
        hi[fit_rows] = new_hi[fit_ok]

        lo_f = lo[fit_rows]
        hi_f = hi[fit_rows]
        npts_f = (hi_f - lo_f).astype(np.float64)

        s1_f = px1[hi_f] - px1[lo_f]
        s2_f = px2[hi_f] - px2[lo_f]
        s3_f = px3[hi_f] - px3[lo_f]
        s4_f = px4[hi_f] - px4[lo_f]
        t0_f = _window_sum(py0, lo_f, hi_f, rows=fit_rows)
        t1_f = _window_sum(py1, lo_f, hi_f, rows=fit_rows)
        t2_f = _window_sum(py2, lo_f, hi_f, rows=fit_rows)

        af = np.empty((fit_rows.size, 3, 3), dtype=np.float64)
        bf = np.empty((fit_rows.size, 3, 1), dtype=np.float64)
        af[:, 0, 0] = s4_f
        af[:, 0, 1] = s3_f
        af[:, 0, 2] = s2_f
        af[:, 1, 0] = s3_f
        af[:, 1, 1] = s2_f
        af[:, 1, 2] = s1_f
        af[:, 2, 0] = s2_f
        af[:, 2, 1] = s1_f
        af[:, 2, 2] = npts_f
        bf[:, 0, 0] = t2_f
        bf[:, 1, 0] = t1_f
        bf[:, 2, 0] = t0_f

        coeffs_f = np.linalg.solve(af, bf)[:, :, 0]
        a_f = coeffs_f[:, 0]
        b_f = coeffs_f[:, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            peak_f = -b_f / (2.0 * a_f)

        ok = np.isfinite(peak_f) & (np.abs(a_f) >= 1e-14)
        if mode == "max":
            ok &= a_f < 0.0
        else:
            ok &= a_f > 0.0

        bad_rows = fit_rows[~ok]
        alive[bad_rows] = False
        can_iter[bad_rows] = False

        good_rows = fit_rows[ok]
        peak_x[good_rows] = peak_f[ok]

    if output_wavelength:
        band_axis = np.arange(wavelengths.size, dtype=np.float64)
        out = np.interp(peak_x, band_axis, wavelengths)
    else:
        out = peak_x

    peaks_flat = peaks.reshape(-1)
    valid_flat = valid.reshape(-1)
    peaks_flat[alive] = out[alive]
    valid_flat[alive] = True
    return peaks, valid


def _select_cubic_extremum(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    *,
    mode: Mode,
    eps: float = 1e-14,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized cubic extremum selection from derivative roots."""
    ok_a = np.abs(a) >= eps
    q = b * b - 3.0 * a * c
    ok_q = q >= 0.0
    sq = np.sqrt(np.maximum(q, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        x1 = (-b - sq) / (3.0 * a)
        x2 = (-b + sq) / (3.0 * a)
    curv1 = 6.0 * a * x1 + 2.0 * b
    curv2 = 6.0 * a * x2 + 2.0 * b
    if mode == "max":
        good1 = ok_a & ok_q & np.isfinite(x1) & (curv1 < 0.0)
        good2 = ok_a & ok_q & np.isfinite(x2) & (curv2 < 0.0)
    else:
        good1 = ok_a & ok_q & np.isfinite(x1) & (curv1 > 0.0)
        good2 = ok_a & ok_q & np.isfinite(x2) & (curv2 > 0.0)

    x = np.where(good1, x1, np.where(good2, x2, np.nan))
    ok = good1 | good2
    return x, ok


def _peak_map_cubic_batched(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    band_start: int,
    band_stop: int,
    *,
    n_iterations: int,
    half_width: int,
    mode: Mode,
    output_wavelength: bool,
    continuum: ContinuumMethod,
    continuum_eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Batched cubic peak/min map with iterative window shifts."""
    m, n = cube.shape[0], cube.shape[1]
    peaks = np.full((m, n), np.nan, dtype=np.float64)
    valid = np.zeros((m, n), dtype=bool)

    x = np.arange(band_start, band_stop, dtype=np.float64)
    k = x.size
    p = m * n
    if k < 5:
        return peaks, valid

    y2d = np.asarray(cube[:, :, band_start:band_stop], dtype=np.float64).reshape(p, k)
    if continuum in ("linear", "hull"):
        y2d = continuum_remove_rows(x, y2d, method=continuum, eps=continuum_eps)

    # Prefix sums for y*x^k, k=0..3
    x1 = x
    x2 = x1 * x1
    x3 = x2 * x1
    x4 = x2 * x2
    x5 = x4 * x1
    x6 = x3 * x3

    py0 = np.pad(np.cumsum(y2d, axis=1), ((0, 0), (1, 0)))
    py1 = np.pad(np.cumsum(y2d * x1[None, :], axis=1), ((0, 0), (1, 0)))
    py2 = np.pad(np.cumsum(y2d * x2[None, :], axis=1), ((0, 0), (1, 0)))
    py3 = np.pad(np.cumsum(y2d * x3[None, :], axis=1), ((0, 0), (1, 0)))

    px1 = np.pad(np.cumsum(x1), (1, 0))
    px2 = np.pad(np.cumsum(x2), (1, 0))
    px3 = np.pad(np.cumsum(x3), (1, 0))
    px4 = np.pad(np.cumsum(x4), (1, 0))
    px5 = np.pad(np.cumsum(x5), (1, 0))
    px6 = np.pad(np.cumsum(x6), (1, 0))

    lo = np.zeros(p, dtype=np.int64)
    hi = np.full(p, k, dtype=np.int64)
    peak_x = np.full(p, np.nan, dtype=np.float64)

    npts = (hi - lo).astype(np.float64)
    s1 = px1[hi] - px1[lo]
    s2 = px2[hi] - px2[lo]
    s3 = px3[hi] - px3[lo]
    s4 = px4[hi] - px4[lo]
    s5 = px5[hi] - px5[lo]
    s6 = px6[hi] - px6[lo]
    t0 = _window_sum(py0, lo, hi)
    t1 = _window_sum(py1, lo, hi)
    t2 = _window_sum(py2, lo, hi)
    t3 = _window_sum(py3, lo, hi)

    a_mat = np.empty((p, 4, 4), dtype=np.float64)
    b_mat = np.empty((p, 4, 1), dtype=np.float64)
    a_mat[:, 0, 0] = s6
    a_mat[:, 0, 1] = s5
    a_mat[:, 0, 2] = s4
    a_mat[:, 0, 3] = s3
    a_mat[:, 1, 0] = s5
    a_mat[:, 1, 1] = s4
    a_mat[:, 1, 2] = s3
    a_mat[:, 1, 3] = s2
    a_mat[:, 2, 0] = s4
    a_mat[:, 2, 1] = s3
    a_mat[:, 2, 2] = s2
    a_mat[:, 2, 3] = s1
    a_mat[:, 3, 0] = s3
    a_mat[:, 3, 1] = s2
    a_mat[:, 3, 2] = s1
    a_mat[:, 3, 3] = npts
    b_mat[:, 0, 0] = t3
    b_mat[:, 1, 0] = t2
    b_mat[:, 2, 0] = t1
    b_mat[:, 3, 0] = t0

    coeffs = np.linalg.solve(a_mat, b_mat)[:, :, 0]
    peak_x, alive = _select_cubic_extremum(coeffs[:, 0], coeffs[:, 1], coeffs[:, 2], mode=mode)

    can_iter = alive.copy()
    for _ in range(1, n_iterations):
        idx_rows = np.where(can_iter)[0]
        if idx_rows.size == 0:
            break

        idx_near = _nearest_int_tie_low(peak_x[idx_rows] - float(band_start))
        lo_i = lo[idx_rows]
        hi_i = hi[idx_rows]
        idx_near = np.clip(idx_near, lo_i, hi_i - 1)

        new_lo = np.maximum(lo_i, idx_near - half_width)
        new_hi = np.minimum(hi_i, idx_near + half_width + 1)
        nnew = new_hi - new_lo

        fit_ok = nnew >= 5
        stop_rows = idx_rows[~fit_ok]
        can_iter[stop_rows] = False

        fit_rows = idx_rows[fit_ok]
        if fit_rows.size == 0:
            continue

        lo[fit_rows] = new_lo[fit_ok]
        hi[fit_rows] = new_hi[fit_ok]

        lo_f = lo[fit_rows]
        hi_f = hi[fit_rows]
        npts_f = (hi_f - lo_f).astype(np.float64)

        s1_f = px1[hi_f] - px1[lo_f]
        s2_f = px2[hi_f] - px2[lo_f]
        s3_f = px3[hi_f] - px3[lo_f]
        s4_f = px4[hi_f] - px4[lo_f]
        s5_f = px5[hi_f] - px5[lo_f]
        s6_f = px6[hi_f] - px6[lo_f]
        t0_f = _window_sum(py0, lo_f, hi_f, rows=fit_rows)
        t1_f = _window_sum(py1, lo_f, hi_f, rows=fit_rows)
        t2_f = _window_sum(py2, lo_f, hi_f, rows=fit_rows)
        t3_f = _window_sum(py3, lo_f, hi_f, rows=fit_rows)

        af = np.empty((fit_rows.size, 4, 4), dtype=np.float64)
        bf = np.empty((fit_rows.size, 4, 1), dtype=np.float64)
        af[:, 0, 0] = s6_f
        af[:, 0, 1] = s5_f
        af[:, 0, 2] = s4_f
        af[:, 0, 3] = s3_f
        af[:, 1, 0] = s5_f
        af[:, 1, 1] = s4_f
        af[:, 1, 2] = s3_f
        af[:, 1, 3] = s2_f
        af[:, 2, 0] = s4_f
        af[:, 2, 1] = s3_f
        af[:, 2, 2] = s2_f
        af[:, 2, 3] = s1_f
        af[:, 3, 0] = s3_f
        af[:, 3, 1] = s2_f
        af[:, 3, 2] = s1_f
        af[:, 3, 3] = npts_f
        bf[:, 0, 0] = t3_f
        bf[:, 1, 0] = t2_f
        bf[:, 2, 0] = t1_f
        bf[:, 3, 0] = t0_f

        coeffs_f = np.linalg.solve(af, bf)[:, :, 0]
        peak_f, ok = _select_cubic_extremum(
            coeffs_f[:, 0],
            coeffs_f[:, 1],
            coeffs_f[:, 2],
            mode=mode,
        )

        bad_rows = fit_rows[~ok]
        alive[bad_rows] = False
        can_iter[bad_rows] = False

        good_rows = fit_rows[ok]
        peak_x[good_rows] = peak_f[ok]

    if output_wavelength:
        band_axis = np.arange(wavelengths.size, dtype=np.float64)
        out = np.interp(peak_x, band_axis, wavelengths)
    else:
        out = peak_x

    peaks_flat = peaks.reshape(-1)
    valid_flat = valid.reshape(-1)
    peaks_flat[alive] = out[alive]
    valid_flat[alive] = True
    return peaks, valid


def load_cube_npy(path: str) -> np.ndarray:
    """Load a numeric array from ``.npy`` (shape ``(M, N, n_bands)``)."""
    arr = np.load(path, allow_pickle=False)
    if not isinstance(arr, np.ndarray):
        raise TypeError("expected ndarray in .npy file")
    if arr.ndim != 3:
        raise ValueError(f"expected 3-D cube, got shape {arr.shape}")
    return arr


def peak_map(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    band_start: int,
    band_stop: int,
    *,
    degree: Literal[2, 3] = 2,
    n_iterations: int = 3,
    half_width: int = 3,
    mode: Mode = "max",
    output_wavelength: bool = True,
    continuum: ContinuumMethod = "none",
    continuum_eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Subsample peak position for every spatial pixel along the spectral axis.

    Parameters
    ----------
    cube
        ``(M, N, n_bands)`` hyperspectral data.
    wavelengths
        ``(n_bands,)`` wavelength per band index (same length as spectral axis).
    band_start, band_stop
        Half-open slice ``[band_start, band_stop)`` defining the fitting window
        (indices into ``wavelengths`` and ``cube[..., :]``).
    degree, n_iterations, half_width, mode
        Passed to :func:`peakfit.refine.refine_peak_subsample`.
    continuum, continuum_eps
        Optional continuum removal on each spectrum before fitting (see
        :func:`peakfit.refine.refine_peak_subsample`).
    output_wavelength
        If True, return peak positions in wavelength units (interpolated);
        if False, return fractional band index.

    Returns
    -------
    peaks, valid
        ``peaks`` has shape ``(M, N)``. ``valid`` is boolean, False where
        refinement failed (filled with ``nan`` in ``peaks``).
    """
    if cube.ndim != 3:
        raise ValueError("cube must have shape (M, N, n_bands)")
    w = np.asarray(wavelengths, dtype=np.float64)
    if w.ndim != 1:
        raise ValueError("wavelengths must be 1-D")
    n_bands = cube.shape[2]
    if w.size != n_bands:
        raise ValueError("wavelengths length must match cube spectral dimension")

    if band_start < 0 or band_stop > n_bands or band_start >= band_stop:
        raise ValueError("invalid band window")

    m, n = cube.shape[0], cube.shape[1]
    peaks = np.full((m, n), np.nan, dtype=np.float64)
    valid = np.zeros((m, n), dtype=bool)

    # Fast vectorized path:
    # quadratic/cubic fit with vectorized continuum removal for none/linear/hull.
    # This computes all pixels in bulk and supports iterative window shifts.
    if (
        degree in (2, 3)
        and continuum in ("none", "linear", "hull")
    ):
        if degree == 2:
            return _peak_map_quadratic_batched(
                cube,
                w,
                band_start,
                band_stop,
                n_iterations=n_iterations,
                half_width=half_width,
                mode=mode,
                output_wavelength=output_wavelength,
                continuum=continuum,
                continuum_eps=continuum_eps,
            )
        return _peak_map_cubic_batched(
            cube,
            w,
            band_start,
            band_stop,
            n_iterations=n_iterations,
            half_width=half_width,
            mode=mode,
            output_wavelength=output_wavelength,
            continuum=continuum,
            continuum_eps=continuum_eps,
        )

    x = np.arange(band_start, band_stop, dtype=np.float64)

    for i in range(m):
        spec_row = cube[i, :, band_start:band_stop]
        for j in range(n):
            y = spec_row[j]
            try:
                res: PeakRefinementResult = refine_peak_subsample(
                    x,
                    y,
                    degree=degree,
                    n_iterations=n_iterations,
                    half_width=half_width,
                    mode=mode,
                    continuum=continuum,
                    continuum_eps=continuum_eps,
                )
            except (ValueError, np.linalg.LinAlgError):
                continue
            px = res.peak_x
            if output_wavelength:
                # res.peak_x is in fractional index units relative to window start
                # x uses band_start, band_stop so peak index in full cube = band_start + (px - band_start)... 
                # Actually x = arange(band_start, band_stop) so values ARE band indices.
                px_global = float(px)
                peaks[i, j] = float(index_to_wavelength(w, px_global))
            else:
                peaks[i, j] = float(px)
            valid[i, j] = True

    return peaks, valid
