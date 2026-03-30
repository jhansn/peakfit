"""Parabolic / polynomial subsample peak estimation for hyperspectral cubes."""

from peakfit.continuum import ContinuumMethod, continuum_remove
from peakfit.cube import load_cube_npy, peak_map
from peakfit.polynomial import (
    ExtremumPoint,
    extrema_from_fit,
    extremum_from_fit,
    fit_polynomial,
    quadratic_vertex,
    cubic_extrema,
)
from peakfit.refine import (
    PeakRefinementResult,
    fit_extrema_subsample,
    refine_peak_subsample,
    subset_around_index,
)
from peakfit.wavelength import (
    index_to_wavelength,
    linear_wavelengths,
    wavelengths_from_step,
)

__all__ = [
    "ContinuumMethod",
    "ExtremumPoint",
    "continuum_remove",
    "cubic_extrema",
    "extrema_from_fit",
    "extremum_from_fit",
    "fit_polynomial",
    "fit_extrema_subsample",
    "index_to_wavelength",
    "linear_wavelengths",
    "load_cube_npy",
    "peak_map",
    "PeakRefinementResult",
    "quadratic_vertex",
    "refine_peak_subsample",
    "subset_around_index",
    "wavelengths_from_step",
]
