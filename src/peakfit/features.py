"""Absorption feature extraction built on top of subsample center refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np

from peakfit.continuum import ContinuumMethod, continuum_remove
from peakfit.refine import PeakRefinementResult, refine_peak_subsample

WindowLike = slice | tuple[int, int]


@dataclass(frozen=True)
class FeatureSupport:
    """Detected feature support and center location."""

    left_x: float
    center_x: float
    right_x: float
    left_idx: int
    center_idx: int
    right_idx: int


@dataclass(frozen=True)
class AbsorptionMetrics:
    """Scalar metrics describing one absorption feature."""

    depth: float
    width: float
    area: float
    left_width: float
    right_width: float
    left_area: float
    right_area: float
    asymmetry_width: float
    asymmetry_area: float
    left_slope: float
    right_slope: float


@dataclass(frozen=True)
class AbsorptionFeature:
    """Full extracted feature payload."""

    support: FeatureSupport
    metrics: AbsorptionMetrics
    descriptor: np.ndarray
    continuum_method: ContinuumMethod
    continuum_removed: np.ndarray
    refinement: PeakRefinementResult


def _validate_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    if x.size < 4:
        raise ValueError("need at least 4 points")
    if np.any(np.diff(x) <= 0):
        raise ValueError("x must be strictly increasing")
    return x, y


def _smooth_support_signal(y: np.ndarray) -> np.ndarray:
    """Small fixed smoothing for shoulder detection only."""
    yp = np.pad(y, 1, mode="edge")
    return (yp[:-2] + yp[1:-1] + yp[2:]) / 3.0


def _is_local_max(y: np.ndarray, idx: int) -> bool:
    return bool(y[idx] >= y[idx - 1] and y[idx] >= y[idx + 1])


def _find_left_support(y: np.ndarray, center_idx: int, boundary_level: float) -> int | None:
    for idx in range(center_idx - 1, 0, -1):
        if _is_local_max(y, idx):
            return idx
        if y[idx] >= boundary_level:
            return idx
    if y[0] >= boundary_level:
        return 0
    return None


def _find_right_support(y: np.ndarray, center_idx: int, boundary_level: float) -> int | None:
    for idx in range(center_idx + 1, y.size - 1):
        if _is_local_max(y, idx):
            return idx
        if y[idx] >= boundary_level:
            return idx
    if y[-1] >= boundary_level:
        return y.size - 1
    return None


def _refine_feature_center(
    x: np.ndarray,
    y: np.ndarray,
    *,
    degree: Literal[2, 3],
    n_iterations: int,
    half_width: int,
) -> PeakRefinementResult:
    last_error: Exception | None = None
    for n_iter in range(n_iterations, 0, -1):
        try:
            return refine_peak_subsample(
                x,
                y,
                degree=degree,
                n_iterations=n_iter,
                half_width=half_width,
                mode="min",
                continuum="none",
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            last_error = exc
    if last_error is None:
        raise ValueError("feature refinement failed")
    raise last_error


def _slice_with_point(
    x: np.ndarray,
    y: np.ndarray,
    x0: float,
    x1: float,
    *,
    extra_x: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    mask = (x > x0) & (x < x1)
    xs = x[mask]
    ys = y[mask]

    points_x = [x0, x1]
    if extra_x is not None and x0 < extra_x < x1:
        points_x.append(extra_x)
    points_x = np.array(sorted(points_x), dtype=np.float64)
    points_y = np.interp(points_x, x, y)

    x_out = np.concatenate([xs, points_x])
    y_out = np.concatenate([ys, points_y])
    order = np.argsort(x_out)
    return x_out[order], y_out[order]


def _integrate_piece(
    x: np.ndarray,
    y: np.ndarray,
    lo: float,
    hi: float,
) -> float:
    xp, yp = _slice_with_point(x, y, lo, hi)
    return float(np.trapezoid(1.0 - yp, xp))


def _build_descriptor(
    x: np.ndarray,
    y: np.ndarray,
    support: FeatureSupport,
    *,
    descriptor_bins: int,
) -> np.ndarray:
    if descriptor_bins < 3:
        raise ValueError("descriptor_bins must be >= 3")

    x_res = np.linspace(support.left_x, support.right_x, descriptor_bins, dtype=np.float64)
    y_res = np.interp(x_res, x, y)

    x_norm = 2.0 * (x_res - support.left_x) / (support.right_x - support.left_x) - 1.0
    y_left = float(np.interp(support.left_x, x, y))
    y_right = float(np.interp(support.right_x, x, y))
    baseline = y_left + (y_right - y_left) * (x_res - support.left_x) / (support.right_x - support.left_x)
    center_depth = float(np.interp(support.center_x, x_res, baseline - y_res))
    if not np.isfinite(center_depth) or center_depth <= 0.0:
        raise ValueError("descriptor normalization requires positive center depth")

    y_norm = (baseline - y_res) / center_depth
    y_norm_s = _smooth_support_signal(y_norm)
    dy_dx = np.gradient(y_norm_s, x_norm)
    return np.arctan(dy_dx).astype(np.float64, copy=False)


def _compute_metrics(
    x: np.ndarray,
    y: np.ndarray,
    support: FeatureSupport,
    *,
    include_area: bool = True,
) -> AbsorptionMetrics:
    y_center = float(np.interp(support.center_x, x, y))
    y_left = float(np.interp(support.left_x, x, y))
    y_right = float(np.interp(support.right_x, x, y))

    depth = 1.0 - y_center
    width = support.right_x - support.left_x
    left_width = support.center_x - support.left_x
    right_width = support.right_x - support.center_x

    asymmetry_width = (right_width - left_width) / width
    if include_area:
        area = _integrate_piece(x, y, support.left_x, support.right_x)
        left_area = _integrate_piece(x, y, support.left_x, support.center_x)
        right_area = _integrate_piece(x, y, support.center_x, support.right_x)
        asymmetry_area = np.nan if area <= 0.0 else (right_area - left_area) / area
    else:
        area = np.nan
        left_area = np.nan
        right_area = np.nan
        asymmetry_area = np.nan

    left_slope = (y_center - y_left) / left_width
    right_slope = (y_right - y_center) / right_width

    return AbsorptionMetrics(
        depth=float(depth),
        width=float(width),
        area=float(area),
        left_width=float(left_width),
        right_width=float(right_width),
        left_area=float(left_area),
        right_area=float(right_area),
        asymmetry_width=float(asymmetry_width),
        asymmetry_area=float(asymmetry_area),
        left_slope=float(left_slope),
        right_slope=float(right_slope),
    )


def _extract_absorption_feature_with_center(
    x: np.ndarray,
    y_cr: np.ndarray,
    *,
    center_x: float,
    continuum_method: ContinuumMethod,
    refinement: PeakRefinementResult,
    boundary_tol: float,
    smooth_support: bool,
    descriptor_bins: int | None,
    include_area: bool = True,
) -> AbsorptionFeature:
    """Build a feature payload from a precomputed center on a CR curve."""
    center_idx = int(np.argmin(np.abs(x - center_x)))
    y_support = _smooth_support_signal(y_cr) if smooth_support else y_cr
    boundary_level = 1.0 - boundary_tol

    left_idx = _find_left_support(y_support, center_idx, boundary_level)
    right_idx = _find_right_support(y_support, center_idx, boundary_level)
    if left_idx is None or right_idx is None:
        raise ValueError("could not determine feature support")
    if left_idx >= center_idx or right_idx <= center_idx:
        raise ValueError("invalid feature support around center")

    support = FeatureSupport(
        left_x=float(x[left_idx]),
        center_x=float(center_x),
        right_x=float(x[right_idx]),
        left_idx=int(left_idx),
        center_idx=int(center_idx),
        right_idx=int(right_idx),
    )
    if not (support.left_x < support.center_x < support.right_x):
        raise ValueError("refined center must lie strictly inside support")

    metrics = _compute_metrics(x, y_cr, support, include_area=include_area)
    if metrics.depth <= 0.0:
        raise ValueError("feature depth must be positive")

    if descriptor_bins is None:
        descriptor = np.empty(0, dtype=np.float64)
    else:
        descriptor = _build_descriptor(x, y_cr, support, descriptor_bins=descriptor_bins)
    return AbsorptionFeature(
        support=support,
        metrics=metrics,
        descriptor=descriptor,
        continuum_method=continuum_method,
        continuum_removed=y_cr,
        refinement=refinement,
    )


def extract_absorption_feature(
    x: np.ndarray,
    y: np.ndarray,
    *,
    degree: Literal[2, 3] = 2,
    continuum: ContinuumMethod = "none",
    continuum_eps: float = 1e-12,
    n_iterations: int = 3,
    half_width: int = 3,
    boundary_tol: float = 0.01,
    smooth_support: bool = True,
    descriptor_bins: int = 16,
) -> AbsorptionFeature:
    """Extract one absorption feature from a pre-windowed spectrum."""
    if not (0.0 <= boundary_tol < 1.0):
        raise ValueError("boundary_tol must be in [0, 1)")

    x, y = _validate_xy(x, y)
    if x.size < degree + 2:
        raise ValueError(f"need at least {degree + 2} points")

    y_cr = (
        continuum_remove(x, y, method=continuum, eps=continuum_eps)
        if continuum != "none"
        else y.copy()
    )

    refinement = _refine_feature_center(
        x,
        y_cr,
        degree=degree,
        n_iterations=n_iterations,
        half_width=half_width,
    )

    return _extract_absorption_feature_with_center(
        x,
        y_cr,
        center_x=float(refinement.peak_x),
        continuum_method=continuum,
        refinement=refinement,
        boundary_tol=boundary_tol,
        smooth_support=smooth_support,
        descriptor_bins=descriptor_bins,
        include_area=True,
    )


def _window_to_indices(window: WindowLike, n: int) -> tuple[int, int]:
    if isinstance(window, slice):
        if window.step not in (None, 1):
            raise ValueError("window slices must use step 1")
        start = 0 if window.start is None else int(window.start)
        stop = n if window.stop is None else int(window.stop)
    else:
        start, stop = int(window[0]), int(window[1])
    if start < 0 or stop > n or start >= stop:
        raise ValueError("invalid window bounds")
    return start, stop


def extract_absorption_features(
    x: np.ndarray,
    y: np.ndarray,
    *,
    windows: Iterable[WindowLike],
    degree: Literal[2, 3] = 2,
    continuum: ContinuumMethod = "none",
    continuum_eps: float = 1e-12,
    n_iterations: int = 3,
    half_width: int = 3,
    boundary_tol: float = 0.01,
    smooth_support: bool = True,
    descriptor_bins: int = 16,
) -> tuple[AbsorptionFeature, ...]:
    """Extract multiple windowed features from one spectrum.

    `windows` are slices or `(start, stop)` index pairs into `x` and `y`.
    """
    x, y = _validate_xy(x, y)
    out: list[AbsorptionFeature] = []
    for window in windows:
        start, stop = _window_to_indices(window, x.size)
        out.append(
            extract_absorption_feature(
                x[start:stop],
                y[start:stop],
                degree=degree,
                continuum=continuum,
                continuum_eps=continuum_eps,
                n_iterations=n_iterations,
                half_width=half_width,
                boundary_tol=boundary_tol,
                smooth_support=smooth_support,
                descriptor_bins=descriptor_bins,
            )
        )
    return tuple(out)
