import numpy as np
import pytest

from peakfit.features import extract_absorption_feature, extract_absorption_features


def _gaussian_trough(
    x: np.ndarray,
    *,
    center: float,
    sigma_left: float,
    sigma_right: float | None = None,
    depth: float = 0.2,
) -> np.ndarray:
    sigma_right = sigma_left if sigma_right is None else sigma_right
    sigmas = np.where(x < center, sigma_left, sigma_right)
    return 1.0 - depth * np.exp(-0.5 * ((x - center) / sigmas) ** 2)


def test_extract_absorption_feature_symmetric_trough_metrics():
    x = np.linspace(-6.0, 6.0, 241)
    y = _gaussian_trough(x, center=0.0, sigma_left=1.1, depth=0.2)

    feat = extract_absorption_feature(
        x,
        y,
        degree=2,
        n_iterations=4,
        half_width=8,
        boundary_tol=0.02,
    )

    assert feat.support.center_x == pytest.approx(0.0, abs=0.06)
    assert feat.metrics.depth == pytest.approx(0.2, abs=0.01)
    assert feat.metrics.width > 4.0
    assert feat.metrics.area > 0.4
    assert feat.metrics.asymmetry_width == pytest.approx(0.0, abs=0.05)
    assert feat.metrics.asymmetry_area == pytest.approx(0.0, abs=0.05)
    assert feat.metrics.left_slope == pytest.approx(-feat.metrics.right_slope, rel=0.08)
    assert feat.descriptor.shape == (16,)
    assert np.all(np.isfinite(feat.descriptor))


def test_extract_absorption_feature_asymmetric_trough_metrics():
    x = np.linspace(-8.0, 8.0, 321)
    y = _gaussian_trough(x, center=0.8, sigma_left=0.9, sigma_right=2.0, depth=0.16)

    feat = extract_absorption_feature(
        x,
        y,
        degree=3,
        n_iterations=4,
        half_width=10,
        boundary_tol=0.015,
    )

    assert feat.support.center_x == pytest.approx(0.8, abs=0.12)
    assert feat.metrics.right_width > feat.metrics.left_width
    assert feat.metrics.right_area > feat.metrics.left_area
    assert feat.metrics.asymmetry_width > 0.15
    assert feat.metrics.asymmetry_area > 0.1


def test_extract_absorption_feature_continuum_linear_and_hull_are_stable():
    x = np.linspace(0.0, 10.0, 301)
    baseline = 0.9 + 0.04 * x
    y = baseline - 0.16 * np.exp(-0.5 * ((x - 5.2) / 0.9) ** 2)

    feat_linear = extract_absorption_feature(
        x,
        y,
        degree=2,
        n_iterations=4,
        half_width=9,
        continuum="linear",
        boundary_tol=0.01,
    )
    feat_hull = extract_absorption_feature(
        x,
        y,
        degree=2,
        n_iterations=4,
        half_width=9,
        continuum="hull",
        boundary_tol=0.01,
    )

    assert feat_linear.support.center_x == pytest.approx(5.2, abs=0.08)
    assert feat_hull.support.center_x == pytest.approx(5.2, abs=0.08)
    assert feat_linear.metrics.depth == pytest.approx(feat_hull.metrics.depth, abs=0.015)
    assert feat_linear.metrics.width == pytest.approx(feat_hull.metrics.width, abs=0.2)
    assert feat_linear.metrics.area == pytest.approx(feat_hull.metrics.area, abs=0.03)


def test_extract_absorption_feature_prefers_local_maxima_shoulders():
    x = np.linspace(-6.0, 6.0, 241)
    y = _gaussian_trough(x, center=0.0, sigma_left=1.2, depth=0.18)
    y += 0.02 * np.exp(-0.5 * ((x + 2.0) / 0.12) ** 2)
    y += 0.02 * np.exp(-0.5 * ((x - 2.0) / 0.12) ** 2)

    feat = extract_absorption_feature(
        x,
        y,
        degree=2,
        n_iterations=4,
        half_width=8,
        boundary_tol=0.005,
        smooth_support=False,
    )

    assert x[feat.support.left_idx] == pytest.approx(-2.0, abs=0.12)
    assert x[feat.support.right_idx] == pytest.approx(2.0, abs=0.12)


def test_extract_absorption_feature_falls_back_to_boundary_crossings():
    x = np.linspace(-6.0, 6.0, 241)
    y = _gaussian_trough(x, center=0.0, sigma_left=1.1, depth=0.2)

    feat = extract_absorption_feature(
        x,
        y,
        degree=2,
        n_iterations=4,
        half_width=8,
        boundary_tol=0.02,
        smooth_support=False,
    )

    level = 1.0 - 0.02
    left_expected = np.max(np.where((x < feat.support.center_x) & (y >= level))[0])
    right_expected = np.min(np.where((x > feat.support.center_x) & (y >= level))[0])
    assert feat.support.left_idx == int(left_expected)
    assert feat.support.right_idx == int(right_expected)


def test_extract_absorption_feature_raises_when_support_is_missing():
    x = np.linspace(-4.0, 4.0, 161)
    y = 0.95 - 0.10 * np.exp(-0.5 * (x / 0.8) ** 2)

    with pytest.raises(ValueError, match="support"):
        extract_absorption_feature(
            x,
            y,
            degree=2,
            n_iterations=4,
            half_width=8,
            boundary_tol=0.01,
            smooth_support=False,
        )


def test_extract_absorption_feature_descriptor_is_stable_under_noise():
    rng = np.random.default_rng(0)
    x = np.linspace(-6.0, 6.0, 241)
    y = _gaussian_trough(x, center=-0.2, sigma_left=1.0, sigma_right=1.6, depth=0.18)
    noisy = y + rng.normal(0.0, 0.003, size=x.size)

    clean_feat = extract_absorption_feature(
        x,
        y,
        degree=3,
        n_iterations=4,
        half_width=8,
        boundary_tol=0.02,
    )
    noisy_feat = extract_absorption_feature(
        x,
        noisy,
        degree=3,
        n_iterations=4,
        half_width=8,
        boundary_tol=0.02,
    )

    cos_sim = float(
        np.dot(clean_feat.descriptor, noisy_feat.descriptor)
        / (np.linalg.norm(clean_feat.descriptor) * np.linalg.norm(noisy_feat.descriptor))
    )
    assert noisy_feat.support.center_x == pytest.approx(clean_feat.support.center_x, abs=0.3)
    assert noisy_feat.metrics.depth == pytest.approx(clean_feat.metrics.depth, abs=0.02)
    assert cos_sim > 0.85


def test_extract_absorption_features_multiple_windows():
    x = np.linspace(0.0, 20.0, 401)
    y = np.ones_like(x)
    y -= 0.12 * np.exp(-0.5 * ((x - 6.0) / 0.8) ** 2)
    y -= 0.09 * np.exp(-0.5 * ((x - 14.0) / 1.1) ** 2)

    features = extract_absorption_features(
        x,
        y,
        windows=[(60, 180), slice(220, 360)],
        degree=2,
        n_iterations=4,
        half_width=8,
        boundary_tol=0.02,
    )

    assert len(features) == 2
    assert features[0].support.center_x == pytest.approx(6.0, abs=0.08)
    assert features[1].support.center_x == pytest.approx(14.0, abs=0.12)
