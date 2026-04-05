import numpy as np
import pytest

from peakfit.cube import (
    SCALAR_FEATURE_NAMES,
    absorption_feature_map,
    peak_map,
)
from peakfit.features import extract_absorption_feature
from peakfit.refine import refine_peak_subsample
from peakfit.wavelength import index_to_wavelength


def _assert_peak_map_matches_reference(
    cube: np.ndarray,
    wl: np.ndarray,
    band_start: int,
    band_stop: int,
    *,
    degree: int,
    n_iterations: int,
    half_width: int,
    mode: str,
    continuum: str,
    output_wavelength: bool,
    atol: float,
) -> None:
    h, w, _ = cube.shape
    p_fast, v_fast = peak_map(
        cube,
        wl,
        band_start,
        band_stop,
        degree=degree,
        n_iterations=n_iterations,
        half_width=half_width,
        mode=mode,
        output_wavelength=output_wavelength,
        continuum=continuum,
    )

    p_ref = np.full((h, w), np.nan, dtype=np.float64)
    v_ref = np.zeros((h, w), dtype=bool)
    x = np.arange(band_start, band_stop, dtype=np.float64)
    for i in range(h):
        for j in range(w):
            try:
                res = refine_peak_subsample(
                    x,
                    cube[i, j, band_start:band_stop],
                    degree=degree,
                    n_iterations=n_iterations,
                    half_width=half_width,
                    mode=mode,
                    continuum=continuum,
                )
            except (ValueError, np.linalg.LinAlgError):
                continue

            if output_wavelength:
                p_ref[i, j] = index_to_wavelength(wl, res.peak_x)
            else:
                p_ref[i, j] = res.peak_x
            v_ref[i, j] = True

    assert np.array_equal(v_fast, v_ref)
    assert np.allclose(p_fast[v_fast], p_ref[v_ref], rtol=0.0, atol=atol)


def test_peak_map_vectorized_matches_generic_none():
    rng = np.random.default_rng(0)
    h, w, b = 5, 6, 30
    wl = np.linspace(1000.0, 2500.0, b)
    cube = rng.normal(0.0, 0.02, size=(h, w, b))
    band_axis = np.arange(b, dtype=np.float64)
    peak = np.exp(-0.5 * ((band_axis - 14.2) / 3.0) ** 2)
    cube += peak[None, None, :]

    _assert_peak_map_matches_reference(
        cube,
        wl,
        7,
        22,
        degree=2,
        n_iterations=1,
        half_width=3,
        mode="max",
        continuum="none",
        output_wavelength=False,
        atol=1e-6,
    )


def test_peak_map_vectorized_matches_generic_linear():
    rng = np.random.default_rng(1)
    h, w, b = 4, 5, 28
    wl = np.linspace(1000.0, 2500.0, b)
    cube = 0.2 + 0.001 * np.arange(b)[None, None, :] + rng.normal(0.0, 0.01, size=(h, w, b))
    trough = np.exp(-0.5 * ((np.arange(b) - 16.5) / 2.5) ** 2)
    cube -= 0.1 * trough[None, None, :]

    _assert_peak_map_matches_reference(
        cube,
        wl,
        10,
        23,
        degree=2,
        n_iterations=1,
        half_width=3,
        mode="min",
        continuum="linear",
        output_wavelength=True,
        atol=1e-6,
    )


def test_peak_map_batched_iterative_matches_generic_none():
    rng = np.random.default_rng(7)
    h, w, b = 5, 7, 34
    wl = np.linspace(1000.0, 2500.0, b)
    cube = rng.normal(0.0, 0.01, size=(h, w, b))
    axis = np.arange(b, dtype=np.float64)
    cube += np.exp(-0.5 * ((axis - 15.4) / 2.8) ** 2)[None, None, :]

    _assert_peak_map_matches_reference(
        cube,
        wl,
        8,
        24,
        degree=2,
        n_iterations=3,
        half_width=3,
        mode="max",
        continuum="none",
        output_wavelength=False,
        atol=1e-8,
    )


def test_peak_map_batched_iterative_matches_generic_linear():
    rng = np.random.default_rng(9)
    h, w, b = 4, 6, 32
    wl = np.linspace(1000.0, 2500.0, b)
    axis = np.arange(b, dtype=np.float64)
    base = 0.4 + 0.003 * axis
    cube = base[None, None, :] + rng.normal(0.0, 0.008, size=(h, w, b))
    cube -= 0.09 * np.exp(-0.5 * ((axis - 18.2) / 2.4) ** 2)[None, None, :]

    _assert_peak_map_matches_reference(
        cube,
        wl,
        11,
        26,
        degree=2,
        n_iterations=3,
        half_width=3,
        mode="min",
        continuum="linear",
        output_wavelength=True,
        atol=1e-8,
    )


def test_peak_map_cubic_batched_matches_generic_none():
    rng = np.random.default_rng(11)
    h, w, b = 4, 5, 40
    wl = np.linspace(1000.0, 2500.0, b)
    axis = np.arange(b, dtype=np.float64)
    cube = 0.25 + rng.normal(0.0, 0.004, size=(h, w, b))
    left = np.exp(-0.5 * ((axis - 22.2) / 2.0) ** 2)
    right = np.exp(-0.5 * ((axis - 22.2) / 4.5) ** 2)
    trough = np.where(axis < 22.2, left, right)
    cube -= 0.08 * trough[None, None, :]

    _assert_peak_map_matches_reference(
        cube,
        wl,
        14,
        31,
        degree=3,
        n_iterations=3,
        half_width=4,
        mode="min",
        continuum="none",
        output_wavelength=False,
        atol=1e-8,
    )


def test_peak_map_cubic_batched_matches_generic_linear():
    rng = np.random.default_rng(13)
    h, w, b = 4, 6, 42
    wl = np.linspace(1000.0, 2500.0, b)
    axis = np.arange(b, dtype=np.float64)
    cube = 0.3 + 0.0015 * axis[None, None, :] + rng.normal(0.0, 0.004, size=(h, w, b))
    left = np.exp(-0.5 * ((axis - 24.0) / 2.2) ** 2)
    right = np.exp(-0.5 * ((axis - 24.0) / 5.2) ** 2)
    trough = np.where(axis < 24.0, left, right)
    cube -= 0.06 * trough[None, None, :]

    _assert_peak_map_matches_reference(
        cube,
        wl,
        16,
        34,
        degree=3,
        n_iterations=3,
        half_width=4,
        mode="min",
        continuum="linear",
        output_wavelength=True,
        atol=1e-6,
    )


def test_peak_map_quadratic_batched_matches_generic_hull():
    rng = np.random.default_rng(17)
    h, w, b = 4, 5, 36
    wl = np.linspace(1000.0, 2500.0, b)
    axis = np.arange(b, dtype=np.float64)
    cube = 0.25 + 0.0018 * axis[None, None, :] + rng.normal(0.0, 0.005, size=(h, w, b))
    cube -= 0.06 * np.exp(-0.5 * ((axis - 19.2) / 2.6) ** 2)[None, None, :]

    _assert_peak_map_matches_reference(
        cube,
        wl,
        12,
        28,
        degree=2,
        n_iterations=3,
        half_width=3,
        mode="min",
        continuum="hull",
        output_wavelength=True,
        atol=1e-6,
    )


def test_peak_map_cubic_batched_matches_generic_hull():
    rng = np.random.default_rng(19)
    h, w, b = 4, 5, 40
    wl = np.linspace(1000.0, 2500.0, b)
    axis = np.arange(b, dtype=np.float64)
    cube = 0.3 + 0.0012 * axis[None, None, :] + rng.normal(0.0, 0.004, size=(h, w, b))
    left = np.exp(-0.5 * ((axis - 23.0) / 2.0) ** 2)
    right = np.exp(-0.5 * ((axis - 23.0) / 4.8) ** 2)
    trough = np.where(axis < 23.0, left, right)
    cube -= 0.07 * trough[None, None, :]

    _assert_peak_map_matches_reference(
        cube,
        wl,
        15,
        33,
        degree=3,
        n_iterations=3,
        half_width=4,
        mode="min",
        continuum="hull",
        output_wavelength=True,
        atol=1e-6,
    )


def test_absorption_feature_map_matches_scalar_reference_linear():
    rng = np.random.default_rng(23)
    h, w, b = 4, 5, 36
    wl = np.linspace(1000.0, 2500.0, b)
    axis = np.arange(b, dtype=np.float64)
    cube = 0.32 + 0.0015 * axis[None, None, :] + rng.normal(0.0, 0.004, size=(h, w, b))
    cube -= 0.07 * np.exp(-0.5 * ((axis - 18.4) / 2.7) ** 2)[None, None, :]

    b0, b1 = 11, 27
    peaks, valid0 = peak_map(
        cube,
        wl,
        b0,
        b1,
        degree=2,
        n_iterations=3,
        half_width=4,
        mode="min",
        output_wavelength=True,
        continuum="linear",
    )
    maps = absorption_feature_map(
        cube,
        wl,
        b0,
        b1,
        degree=2,
        n_iterations=3,
        half_width=4,
        continuum="linear",
        feature_names=SCALAR_FEATURE_NAMES,
        peak_positions=peaks,
        valid_mask=valid0,
    )

    valid = maps.valid
    assert set(maps.keys()) == set(SCALAR_FEATURE_NAMES)

    x = wl[b0:b1]
    for i in range(h):
        for j in range(w):
            if not valid[i, j]:
                assert np.isnan(maps["depth"][i, j])
                continue
            feat = extract_absorption_feature(
                x,
                cube[i, j, b0:b1],
                degree=2,
                n_iterations=3,
                half_width=4,
                continuum="linear",
            )
            assert maps["center"][i, j] == pytest.approx(feat.support.center_x, abs=1e-8)
            assert maps["depth"][i, j] == pytest.approx(feat.metrics.depth, abs=1e-8)
            assert maps["width"][i, j] == pytest.approx(feat.metrics.width, abs=1e-8)
            assert maps["area"][i, j] == pytest.approx(feat.metrics.area, abs=1e-8)
            assert maps["asymmetry_area"][i, j] == pytest.approx(feat.metrics.asymmetry_area, abs=1e-8)


def test_absorption_feature_map_output_index_units():
    rng = np.random.default_rng(29)
    h, w, b = 3, 4, 34
    wl = np.linspace(1000.0, 2500.0, b)
    axis = np.arange(b, dtype=np.float64)
    cube = 1.0 + rng.normal(0.0, 0.003, size=(h, w, b))
    cube -= 0.06 * np.exp(-0.5 * ((axis - 16.1) / 2.5) ** 2)[None, None, :]

    b0, b1 = 9, 24
    peaks, valid0 = peak_map(
        cube,
        wl,
        b0,
        b1,
        degree=2,
        n_iterations=3,
        half_width=4,
        mode="min",
        output_wavelength=False,
        continuum="none",
    )
    maps = absorption_feature_map(
        cube,
        wl,
        b0,
        b1,
        degree=2,
        n_iterations=3,
        half_width=4,
        continuum="none",
        output_wavelength=False,
        peak_positions=peaks,
        valid_mask=valid0,
    )

    valid = maps.valid
    assert np.array_equal(valid, valid0)
    assert np.allclose(maps["center"][valid], peaks[valid], atol=1e-8, rtol=0.0)


def test_absorption_feature_map_defaults_to_full_subset():
    rng = np.random.default_rng(31)
    h, w, b = 3, 4, 30
    wl = np.linspace(1000.0, 2500.0, b)
    axis = np.arange(b, dtype=np.float64)
    cube = 1.0 + rng.normal(0.0, 0.003, size=(h, w, b))
    cube -= 0.05 * np.exp(-0.5 * ((axis - 15.0) / 2.3) ** 2)[None, None, :]

    maps = absorption_feature_map(
        cube,
        wl,
        8,
        23,
        degree=2,
        n_iterations=3,
        half_width=4,
        continuum="none",
    )

    assert set(maps.keys()) == set(SCALAR_FEATURE_NAMES)
    assert maps.valid.shape == (h, w)


def test_absorption_feature_maps_supports_attribute_and_dict_access():
    rng = np.random.default_rng(37)
    h, w, b = 3, 4, 30
    wl = np.linspace(1000.0, 2500.0, b)
    axis = np.arange(b, dtype=np.float64)
    cube = 1.0 + rng.normal(0.0, 0.003, size=(h, w, b))
    cube -= 0.05 * np.exp(-0.5 * ((axis - 15.0) / 2.3) ** 2)[None, None, :]

    maps = absorption_feature_map(
        cube,
        wl,
        8,
        23,
        degree=2,
        n_iterations=3,
        half_width=4,
        continuum="none",
        feature_names=("depth", "width"),
    )

    assert np.array_equal(maps.depth, maps["depth"])
    assert np.array_equal(maps.width, maps["width"])
    assert maps.area is None
    assert set(maps.as_dict()) == {"depth", "width"}
