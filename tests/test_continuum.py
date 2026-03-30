import numpy as np
import pytest

from peakfit.continuum import continuum_remove, continuum_remove_rows


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


def test_continuum_remove_rows_hull_matches_scalar():
    rng = np.random.default_rng(3)
    x = np.linspace(1000.0, 1200.0, 25)
    y2d = 0.3 + rng.normal(0.0, 0.01, size=(6, x.size))
    y2d -= 0.04 * np.exp(-0.5 * ((np.arange(x.size) - 12.3) / 2.2) ** 2)[None, :]
    out_rows = continuum_remove_rows(x, y2d, method="hull")
    out_ref = np.vstack([continuum_remove(x, y2d[i], method="hull") for i in range(y2d.shape[0])])
    assert np.allclose(out_rows, out_ref, rtol=0.0, atol=1e-10)
