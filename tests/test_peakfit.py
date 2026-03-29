import numpy as np
import pytest

from peakfit.polynomial import (
    cubic_extrema,
    extremum_from_fit,
    fit_polynomial,
    quadratic_vertex,
)
from peakfit.continuum import continuum_remove
from peakfit.cube import peak_map
from peakfit.refine import refine_peak_subsample
from peakfit.wavelength import index_to_wavelength, linear_wavelengths


def test_quadratic_vertex_analytic():
    # -(x-2)^2 + 1 = -x^2 + 4x - 3  => vertex at x=2
    coeffs = np.array([-1.0, 4.0, -3.0])
    assert quadratic_vertex(coeffs) == pytest.approx(2.0)


def test_refine_parabolic_peak_max():
    x = np.arange(0.0, 10.0)
    y = -(x - 4.2) ** 2
    res = refine_peak_subsample(
        x, y, degree=2, n_iterations=4, half_width=2, mode="max"
    )
    assert res.peak_x == pytest.approx(4.2, abs=0.05)
    assert res.converged


def test_index_to_wavelength_linear():
    w = linear_wavelengths(5, 400.0, 500.0)
    assert index_to_wavelength(w, 0.0) == pytest.approx(400.0)
    assert index_to_wavelength(w, 4.0) == pytest.approx(500.0)
    assert index_to_wavelength(w, 2.0) == pytest.approx(450.0)
    assert index_to_wavelength(w, 2.5) == pytest.approx(462.5)


def test_cubic_extrema_simple():
    # y = -x^3 + 3x  => y' = -3x^2 + 3 = 0 => x = ±1 ; max at x=1, min at x=-1
    coeffs = np.array([-1.0, 0.0, 3.0, 0.0])
    xm, xn = cubic_extrema(coeffs)
    assert xm == pytest.approx(1.0)
    assert xn == pytest.approx(-1.0)
    assert extremum_from_fit(coeffs, 3, "max") == pytest.approx(1.0)
    assert extremum_from_fit(coeffs, 3, "min") == pytest.approx(-1.0)


def test_continuum_remove_linear_flat():
    x = np.linspace(0.0, 1.0, 40)
    y = 0.2 + 0.5 * x
    cr = continuum_remove(x, y, method="linear")
    assert cr == pytest.approx(1.0, abs=1e-9)


def test_continuum_remove_hull_endpoints_one():
    x = np.array([0.0, 0.5, 1.0])
    y = np.array([1.0, 0.5, 1.0])
    cr = continuum_remove(x, y, method="hull")
    assert cr[0] == pytest.approx(1.0)
    assert cr[2] == pytest.approx(1.0)
    assert cr[1] < 1.0


def test_refine_with_continuum_linear():
    x = np.linspace(0.0, 10.0, 80)
    y = 0.25 + 10.0 * np.exp(-0.5 * (x - 5.0) ** 2)
    res = refine_peak_subsample(
        x, y, degree=2, n_iterations=4, half_width=4, mode="max", continuum="linear"
    )
    assert res.peak_x == pytest.approx(5.0, abs=0.15)


def test_fit_polynomial_degrees():
    x = np.linspace(-1, 1, 10)
    y = x**2
    c2 = fit_polynomial(x, y, 2)
    assert c2[0] == pytest.approx(1.0, abs=0.05)
    c3 = fit_polynomial(x, y, 3)
    assert c3[0] == pytest.approx(0.0, abs=0.1)


def test_peak_map_vectorized_matches_generic_none():
    rng = np.random.default_rng(0)
    h, w, b = 5, 6, 30
    wl = np.linspace(1000.0, 2500.0, b)
    cube = rng.normal(0.0, 0.02, size=(h, w, b))
    # Add smooth peak near band-center to make maxima well-defined.
    band_axis = np.arange(b, dtype=np.float64)
    peak = np.exp(-0.5 * ((band_axis - 14.2) / 3.0) ** 2)
    cube += peak[None, None, :]

    # Vectorized path
    p_fast, v_fast = peak_map(
        cube,
        wl,
        7,
        22,
        degree=2,
        n_iterations=1,
        half_width=3,
        mode="max",
        output_wavelength=False,
        continuum="none",
    )

    # Manual per-pixel baseline via refine_peak_subsample.
    from peakfit.refine import refine_peak_subsample

    p_ref = np.full((h, w), np.nan, dtype=np.float64)
    v_ref = np.zeros((h, w), dtype=bool)
    x = np.arange(7, 22, dtype=np.float64)
    for i in range(h):
        for j in range(w):
            try:
                res = refine_peak_subsample(
                    x,
                    cube[i, j, 7:22],
                    degree=2,
                    n_iterations=1,
                    half_width=3,
                    mode="max",
                    continuum="none",
                )
            except (ValueError, np.linalg.LinAlgError):
                continue
            p_ref[i, j] = res.peak_x
            v_ref[i, j] = True

    assert np.array_equal(v_fast, v_ref)
    assert np.allclose(p_fast[v_fast], p_ref[v_ref], rtol=0.0, atol=1e-8)


def test_peak_map_vectorized_matches_generic_linear():
    rng = np.random.default_rng(1)
    h, w, b = 4, 5, 28
    wl = np.linspace(1000.0, 2500.0, b)
    cube = 0.2 + 0.001 * np.arange(b)[None, None, :] + rng.normal(0.0, 0.01, size=(h, w, b))
    trough = np.exp(-0.5 * ((np.arange(b) - 16.5) / 2.5) ** 2)
    cube -= 0.1 * trough[None, None, :]

    p_fast, v_fast = peak_map(
        cube,
        wl,
        10,
        23,
        degree=2,
        n_iterations=1,
        half_width=3,
        mode="min",
        output_wavelength=True,
        continuum="linear",
    )

    from peakfit.refine import refine_peak_subsample

    p_ref = np.full((h, w), np.nan, dtype=np.float64)
    v_ref = np.zeros((h, w), dtype=bool)
    x = np.arange(10, 23, dtype=np.float64)
    for i in range(h):
        for j in range(w):
            try:
                res = refine_peak_subsample(
                    x,
                    cube[i, j, 10:23],
                    degree=2,
                    n_iterations=1,
                    half_width=3,
                    mode="min",
                    continuum="linear",
                )
            except (ValueError, np.linalg.LinAlgError):
                continue
            p_ref[i, j] = index_to_wavelength(wl, res.peak_x)
            v_ref[i, j] = True

    assert np.array_equal(v_fast, v_ref)
    assert np.allclose(p_fast[v_fast], p_ref[v_ref], rtol=0.0, atol=1e-8)
