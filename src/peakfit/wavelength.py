"""Band index ↔ wavelength mapping for hyperspectral data."""

from __future__ import annotations

import numpy as np


def linear_wavelengths(n_bands: int, start_nm: float, end_nm: float) -> np.ndarray:
    """Uniformly spaced wavelengths for `n_bands` (inclusive endpoints)."""
    if n_bands < 2:
        raise ValueError("n_bands must be at least 2")
    return np.linspace(start_nm, end_nm, n_bands, dtype=np.float64)


def wavelengths_from_step(start_nm: float, step_nm: float, n_bands: int) -> np.ndarray:
    """Arithmetic sequence: start_nm + k * step_nm for k = 0 .. n_bands-1."""
    if n_bands < 1:
        raise ValueError("n_bands must be at least 1")
    return start_nm + step_nm * np.arange(n_bands, dtype=np.float64)


def index_to_wavelength(
    wavelengths: np.ndarray,
    index: float | np.ndarray,
) -> float | np.ndarray:
    """Map fractional band index to wavelength via linear interpolation.

    `wavelengths` has shape ``(n_bands,)``. ``index`` may be fractional
    (sub-band peak position). Uses ``numpy.interp`` over ``0 .. n_bands-1``.
    """
    w = np.asarray(wavelengths, dtype=np.float64)
    if w.ndim != 1:
        raise ValueError("wavelengths must be 1-D")
    n = w.size
    if n < 2:
        raise ValueError("wavelengths must have at least 2 points")
    grid = np.arange(n, dtype=np.float64)
    return np.interp(index, grid, w)
