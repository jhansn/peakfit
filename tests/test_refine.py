import numpy as np
import pytest

from peakfit.refine import fit_extrema_subsample, refine_peak_subsample


def test_refine_parabolic_peak_max():
    x = np.arange(0.0, 10.0)
    y = -(x - 4.2) ** 2
    res = refine_peak_subsample(
        x, y, degree=2, n_iterations=4, half_width=2, mode="max"
    )
    assert res.peak_x == pytest.approx(4.2, abs=0.05)
    assert res.converged


def test_refine_adaptive_window_converges_with_fewer_steps():
    x = np.arange(0.0, 25.0)
    y = -(x - 12.3) ** 2
    res = refine_peak_subsample(
        x,
        y,
        degree=2,
        n_iterations=8,
        half_width=8,
        adaptive_window=True,
        half_width_min=3,
        half_width_shrink=0.65,
        mode="max",
        atol=0.0,
    )
    assert res.peak_x == pytest.approx(12.3, abs=0.05)
    assert res.converged
    assert res.iterations < 8


def test_fit_extrema_subsample_mode_free():
    x = np.linspace(-2.0, 2.0, 25)
    y = (x - 0.4) ** 2 + 0.2
    ex = fit_extrema_subsample(x, y, degree=2, continuum="none")
    assert len(ex) == 1
    assert ex[0].kind == "min"
    assert ex[0].x == pytest.approx(0.4, abs=0.03)


def test_refine_with_continuum_linear():
    x = np.linspace(0.0, 10.0, 80)
    y = 0.25 + 10.0 * np.exp(-0.5 * (x - 5.0) ** 2)
    res = refine_peak_subsample(
        x, y, degree=2, n_iterations=4, half_width=4, mode="max", continuum="linear"
    )
    assert res.peak_x == pytest.approx(5.0, abs=0.15)
