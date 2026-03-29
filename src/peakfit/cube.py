"""Hyperspectral cube I/O and per-pixel peak mapping."""

from __future__ import annotations

from typing import Literal

import numpy as np

from peakfit.continuum import ContinuumMethod
from peakfit.polynomial import Mode
from peakfit.refine import PeakRefinementResult, refine_peak_subsample
from peakfit.wavelength import index_to_wavelength


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
    # quadratic fit, single iteration, and either no continuum or linear continuum.
    # This computes all pixels in bulk via a precomputed pseudoinverse instead of
    # per-pixel np.polyfit calls.
    if (
        degree == 2
        and n_iterations == 1
        and continuum in ("none", "linear")
    ):
        x = np.arange(band_start, band_stop, dtype=np.float64)
        k = x.size
        y2d = np.asarray(cube[:, :, band_start:band_stop], dtype=np.float64).reshape(-1, k)

        if continuum == "linear":
            x0, x1 = x[0], x[-1]
            t = ((x - x0) / (x1 - x0)).reshape(1, -1)
            y0 = y2d[:, [0]]
            y1 = y2d[:, [-1]]
            cont = y0 + (y1 - y0) * t
            y2d = y2d / np.maximum(cont, continuum_eps)

        # Fit y ~ a*x^2 + b*x + c
        design = np.column_stack([x * x, x, np.ones_like(x)])
        pinv = np.linalg.pinv(design)  # (3, k)
        coeffs = y2d @ pinv.T  # (m*n, 3)
        a = coeffs[:, 0]
        b = coeffs[:, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            xv = -b / (2.0 * a)

        if mode == "max":
            ok = a < 0.0
        else:
            ok = a > 0.0
        ok &= np.isfinite(xv)
        ok &= np.abs(a) >= 1e-14

        if output_wavelength:
            band_axis = np.arange(n_bands, dtype=np.float64)
            px = np.interp(xv, band_axis, w)
        else:
            px = xv

        peaks_flat = peaks.reshape(-1)
        valid_flat = valid.reshape(-1)
        peaks_flat[ok] = px[ok]
        valid_flat[ok] = True
        return peaks, valid

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
