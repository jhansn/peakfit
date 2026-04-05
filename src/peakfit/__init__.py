"""Parabolic / polynomial subsample peak estimation for hyperspectral cubes."""

from peakfit.continuum import ContinuumMethod, continuum_remove
from peakfit.cube import (
    AbsorptionFeatureMaps,
    FAST_SCALAR_FEATURE_NAMES,
    SCALAR_FEATURE_NAMES,
    absorption_feature_map,
    load_cube_npy,
    peak_map,
)
from peakfit.features import (
    AbsorptionFeature,
    AbsorptionMetrics,
    FeatureSupport,
    extract_absorption_feature,
    extract_absorption_features,
)
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
    "AbsorptionFeature",
    "AbsorptionFeatureMaps",
    "AbsorptionMetrics",
    "absorption_feature_map",
    "ContinuumMethod",
    "ExtremumPoint",
    "FeatureSupport",
    "continuum_remove",
    "cubic_extrema",
    "extrema_from_fit",
    "extremum_from_fit",
    "extract_absorption_feature",
    "extract_absorption_features",
    "FAST_SCALAR_FEATURE_NAMES",
    "fit_polynomial",
    "fit_extrema_subsample",
    "index_to_wavelength",
    "linear_wavelengths",
    "load_cube_npy",
    "peak_map",
    "PeakRefinementResult",
    "quadratic_vertex",
    "refine_peak_subsample",
    "SCALAR_FEATURE_NAMES",
    "subset_around_index",
    "wavelengths_from_step",
]
