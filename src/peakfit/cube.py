"""Hyperspectral cube I/O, peak mapping, and scalar feature maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from peakfit.continuum import ContinuumMethod, continuum_remove_rows
from peakfit.polynomial import Mode
from peakfit.refine import PeakRefinementResult, refine_peak_subsample
from peakfit.wavelength import index_to_wavelength

ScalarFeatureName = Literal[
    "center",
    "depth",
    "width",
    "area",
    "left_width",
    "right_width",
    "left_area",
    "right_area",
    "asymmetry_width",
    "asymmetry_area",
    "left_slope",
    "right_slope",
]

FAST_SCALAR_FEATURE_NAMES: tuple[ScalarFeatureName, ...] = (
    "center",
    "depth",
    "width",
    "left_width",
    "right_width",
    "asymmetry_width",
    "left_slope",
    "right_slope",
)

SCALAR_FEATURE_NAMES: tuple[ScalarFeatureName, ...] = (
    "center",
    "depth",
    "width",
    "area",
    "left_width",
    "right_width",
    "left_area",
    "right_area",
    "asymmetry_width",
    "asymmetry_area",
    "left_slope",
    "right_slope",
)


@dataclass(frozen=True)
class AbsorptionFeatureMaps:
    """Scalar absorption-feature maps with attribute access."""

    valid: np.ndarray
    center: np.ndarray | None = None
    depth: np.ndarray | None = None
    width: np.ndarray | None = None
    area: np.ndarray | None = None
    left_width: np.ndarray | None = None
    right_width: np.ndarray | None = None
    left_area: np.ndarray | None = None
    right_area: np.ndarray | None = None
    asymmetry_width: np.ndarray | None = None
    asymmetry_area: np.ndarray | None = None
    left_slope: np.ndarray | None = None
    right_slope: np.ndarray | None = None

    def as_dict(self) -> dict[str, np.ndarray]:
        """Return only the populated scalar feature maps."""
        return {
            name: value
            for name in SCALAR_FEATURE_NAMES
            if (value := getattr(self, name)) is not None
        }

    def __getitem__(self, name: ScalarFeatureName) -> np.ndarray:
        value = getattr(self, name)
        if value is None:
            raise KeyError(name)
        return value

    def keys(self) -> tuple[str, ...]:
        return tuple(self.as_dict().keys())

    def items(self):
        return self.as_dict().items()

    def values(self):
        return self.as_dict().values()


def _smooth_rows3(y2d: np.ndarray) -> np.ndarray:
    yp = np.pad(y2d, ((0, 0), (1, 1)), mode="edge")
    return (yp[:, :-2] + yp[:, 1:-1] + yp[:, 2:]) / 3.0


def _nearest_indices_on_axis(x: np.ndarray, points: np.ndarray) -> np.ndarray:
    hi = np.searchsorted(x, points, side="left")
    hi = np.clip(hi, 1, x.size - 1)
    lo = hi - 1
    d_lo = np.abs(points - x[lo])
    d_hi = np.abs(x[hi] - points)
    return np.where(d_hi < d_lo, hi, lo).astype(np.int64)


def _interp_rows_at_points(x: np.ndarray, y2d: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hi = np.searchsorted(x, points, side="left")
    hi = np.clip(hi, 1, x.size - 1)
    lo = hi - 1
    x0 = x[lo]
    x1 = x[hi]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (points - x0) / (x1 - x0)
    rows = np.arange(y2d.shape[0], dtype=np.intp)
    y0 = y2d[rows, lo]
    y1 = y2d[rows, hi]
    yv = y0 + t * (y1 - y0)
    return yv, lo, hi


def _find_support_indices_batch(
    y_support: np.ndarray,
    center_idx: np.ndarray,
    boundary_level: float,
) -> tuple[np.ndarray, np.ndarray]:
    p, k = y_support.shape
    local_max = np.zeros((p, k), dtype=bool)
    if k >= 3:
        local_max[:, 1:-1] = (y_support[:, 1:-1] >= y_support[:, :-2]) & (y_support[:, 1:-1] >= y_support[:, 2:])

    rows = np.arange(p, dtype=np.intp)
    left_idx = np.full(p, -1, dtype=np.int64)
    right_idx = np.full(p, -1, dtype=np.int64)
    left_done = np.zeros(p, dtype=bool)
    right_done = np.zeros(p, dtype=bool)

    for off in range(1, k):
        li = center_idx - off
        active_l = (~left_done) & (li >= 1)
        if np.any(active_l):
            cand_l = active_l & (local_max[rows, np.clip(li, 0, k - 1)] | (y_support[rows, np.clip(li, 0, k - 1)] >= boundary_level))
            left_idx[cand_l] = li[cand_l]
            left_done[cand_l] = True

        ri = center_idx + off
        active_r = (~right_done) & (ri <= (k - 2))
        if np.any(active_r):
            cand_r = active_r & (local_max[rows, np.clip(ri, 0, k - 1)] | (y_support[rows, np.clip(ri, 0, k - 1)] >= boundary_level))
            right_idx[cand_r] = ri[cand_r]
            right_done[cand_r] = True

        if np.all(left_done & right_done):
            break

    left_edge = (~left_done) & (y_support[:, 0] >= boundary_level)
    right_edge = (~right_done) & (y_support[:, -1] >= boundary_level)
    left_idx[left_edge] = 0
    right_idx[right_edge] = k - 1
    return left_idx, right_idx


def _prefix_trapezoid_areas(x: np.ndarray, y2d: np.ndarray) -> np.ndarray:
    dx = np.diff(x)
    seg = 0.5 * ((1.0 - y2d[:, :-1]) + (1.0 - y2d[:, 1:])) * dx[None, :]
    return np.pad(np.cumsum(seg, axis=1), ((0, 0), (1, 0)))


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


def absorption_feature_map(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    band_start: int,
    band_stop: int,
    *,
    degree: Literal[2, 3] = 2,
    n_iterations: int = 3,
    half_width: int = 3,
    continuum: ContinuumMethod = "none",
    continuum_eps: float = 1e-12,
    boundary_tol: float = 0.01,
    smooth_support: bool = True,
    output_wavelength: bool = True,
    feature_names: tuple[ScalarFeatureName, ...] = SCALAR_FEATURE_NAMES,
    peak_positions: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
) -> AbsorptionFeatureMaps:
    """Scalar absorption-feature maps for every valid pixel in a spectral window.

    This reuses minima from :func:`peak_map` when `peak_positions` and `valid_mask`
    are provided, otherwise it computes them internally.
    """
    if cube.ndim != 3:
        raise ValueError("cube must have shape (M, N, n_bands)")
    requested = tuple(dict.fromkeys(feature_names))
    invalid = [name for name in requested if name not in SCALAR_FEATURE_NAMES]
    if invalid:
        raise ValueError(f"unknown feature names: {invalid}")
    if not requested:
        raise ValueError("feature_names must not be empty")
    w = np.asarray(wavelengths, dtype=np.float64)
    if w.ndim != 1:
        raise ValueError("wavelengths must be 1-D")
    n_bands = cube.shape[2]
    if w.size != n_bands:
        raise ValueError("wavelengths length must match cube spectral dimension")
    if band_start < 0 or band_stop > n_bands or band_start >= band_stop:
        raise ValueError("invalid band window")

    m, n = cube.shape[0], cube.shape[1]
    if peak_positions is None or valid_mask is None:
        peak_positions, valid_mask = peak_map(
            cube,
            w,
            band_start,
            band_stop,
            degree=degree,
            n_iterations=n_iterations,
            half_width=half_width,
            mode="min",
            output_wavelength=output_wavelength,
            continuum=continuum,
            continuum_eps=continuum_eps,
        )
    else:
        peak_positions = np.asarray(peak_positions, dtype=np.float64)
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if peak_positions.shape != (m, n) or valid_mask.shape != (m, n):
            raise ValueError("peak_positions and valid_mask must match cube spatial shape")

    x = w[band_start:band_stop] if output_wavelength else np.arange(band_start, band_stop, dtype=np.float64)
    y2d = np.asarray(cube[:, :, band_start:band_stop], dtype=np.float64).reshape(m * n, -1)
    if continuum == "none":
        y2d_cr = y2d.copy()
    else:
        y2d_cr = continuum_remove_rows(x, y2d, method=continuum, eps=continuum_eps)

    peak_flat = np.asarray(peak_positions, dtype=np.float64).reshape(-1)
    valid_seed = np.asarray(valid_mask, dtype=bool).reshape(-1) & np.isfinite(peak_flat)
    include_area = any(name in {"area", "left_area", "right_area", "asymmetry_area"} for name in requested)

    out = {name: np.full(m * n, np.nan, dtype=np.float64) for name in requested}
    valid = np.zeros(m * n, dtype=bool)

    rows = np.flatnonzero(valid_seed)
    if rows.size == 0:
        return AbsorptionFeatureMaps(valid=valid.reshape(m, n), **{name: arr.reshape(m, n) for name, arr in out.items()})

    y_rows = y2d_cr[rows]
    px = peak_flat[rows]
    in_bounds = (px >= float(x[0])) & (px <= float(x[-1]))
    if not np.all(in_bounds):
        rows = rows[in_bounds]
        y_rows = y_rows[in_bounds]
        px = px[in_bounds]
    if rows.size == 0:
        return AbsorptionFeatureMaps(valid=valid.reshape(m, n), **{name: arr.reshape(m, n) for name, arr in out.items()})

    center_idx = _nearest_indices_on_axis(x, px)
    y_support = _smooth_rows3(y_rows) if smooth_support else y_rows
    left_idx, right_idx = _find_support_indices_batch(y_support, center_idx, 1.0 - boundary_tol)

    ok = (left_idx >= 0) & (right_idx >= 0) & (left_idx < center_idx) & (center_idx < right_idx)
    if not np.any(ok):
        return AbsorptionFeatureMaps(valid=valid.reshape(m, n), **{name: arr.reshape(m, n) for name, arr in out.items()})

    rows = rows[ok]
    y_rows = y_rows[ok]
    px = px[ok]
    center_idx = center_idx[ok]
    left_idx = left_idx[ok]
    right_idx = right_idx[ok]

    left_x = x[left_idx]
    right_x = x[right_idx]
    inside = (left_x < px) & (px < right_x)
    if not np.all(inside):
        rows = rows[inside]
        y_rows = y_rows[inside]
        px = px[inside]
        center_idx = center_idx[inside]
        left_idx = left_idx[inside]
        right_idx = right_idx[inside]
        left_x = left_x[inside]
        right_x = right_x[inside]
    if rows.size == 0:
        return AbsorptionFeatureMaps(valid=valid.reshape(m, n), **{name: arr.reshape(m, n) for name, arr in out.items()})

    y_center, lo_idx, _ = _interp_rows_at_points(x, y_rows, px)
    p_rows = np.arange(rows.size, dtype=np.intp)
    y_left = y_rows[p_rows, left_idx]
    y_right = y_rows[p_rows, right_idx]

    depth = 1.0 - y_center
    width = right_x - left_x
    left_width = px - left_x
    right_width = right_x - px
    asymmetry_width = (right_width - left_width) / width
    left_slope = (y_center - y_left) / left_width
    right_slope = (y_right - y_center) / right_width

    metric_ok = np.isfinite(depth) & (depth > 0.0) & np.isfinite(width) & (width > 0.0)
    metric_ok &= np.isfinite(left_width) & (left_width > 0.0) & np.isfinite(right_width) & (right_width > 0.0)
    metric_ok &= np.isfinite(left_slope) & np.isfinite(right_slope) & np.isfinite(asymmetry_width)

    area = left_area = right_area = asymmetry_area = None
    if include_area:
        prefix = _prefix_trapezoid_areas(x, y_rows)
        total_area = prefix[p_rows, right_idx] - prefix[p_rows, left_idx]
        x_lo = x[lo_idx]
        y_lo = y_rows[p_rows, lo_idx]
        partial = 0.5 * ((1.0 - y_lo) + (1.0 - y_center)) * (px - x_lo)
        left_area_v = prefix[p_rows, lo_idx] - prefix[p_rows, left_idx] + partial
        right_area_v = total_area - left_area_v
        with np.errstate(divide="ignore", invalid="ignore"):
            asymmetry_area_v = (right_area_v - left_area_v) / total_area
        metric_ok &= np.isfinite(total_area) & (total_area > 0.0)
        metric_ok &= np.isfinite(left_area_v) & np.isfinite(right_area_v) & np.isfinite(asymmetry_area_v)
        area = total_area
        left_area = left_area_v
        right_area = right_area_v
        asymmetry_area = asymmetry_area_v

    if not np.any(metric_ok):
        return {name: arr.reshape(m, n) for name, arr in out.items()}, valid.reshape(m, n)

    rows = rows[metric_ok]
    px = px[metric_ok]
    depth = depth[metric_ok]
    width = width[metric_ok]
    left_width = left_width[metric_ok]
    right_width = right_width[metric_ok]
    asymmetry_width = asymmetry_width[metric_ok]
    left_slope = left_slope[metric_ok]
    right_slope = right_slope[metric_ok]
    if include_area:
        area = area[metric_ok]
        left_area = left_area[metric_ok]
        right_area = right_area[metric_ok]
        asymmetry_area = asymmetry_area[metric_ok]

    value_map: dict[str, np.ndarray] = {
        "center": px,
        "depth": depth,
        "width": width,
        "left_width": left_width,
        "right_width": right_width,
        "asymmetry_width": asymmetry_width,
        "left_slope": left_slope,
        "right_slope": right_slope,
    }
    if include_area:
        value_map["area"] = area
        value_map["left_area"] = left_area
        value_map["right_area"] = right_area
        value_map["asymmetry_area"] = asymmetry_area

    for name in requested:
        out[name][rows] = value_map[name]
    valid[rows] = True

    return AbsorptionFeatureMaps(
        valid=valid.reshape(m, n),
        **{name: arr.reshape(m, n) for name, arr in out.items()},
    )
