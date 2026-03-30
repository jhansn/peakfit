import numpy as np
import pytest

from peakfit.polynomial import (
    cubic_extrema,
    extrema_from_fit,
    extremum_from_fit,
    fit_polynomial,
    quadratic_vertex,
)


def test_quadratic_vertex_analytic():
    # -(x-2)^2 + 1 = -x^2 + 4x - 3  => vertex at x=2
    coeffs = np.array([-1.0, 4.0, -3.0])
    assert quadratic_vertex(coeffs) == pytest.approx(2.0)


def test_cubic_extrema_simple():
    # y = -x^3 + 3x  => y' = -3x^2 + 3 = 0 => x = ±1 ; max at x=1, min at x=-1
    coeffs = np.array([-1.0, 0.0, 3.0, 0.0])
    xm, xn = cubic_extrema(coeffs)
    assert xm == pytest.approx(1.0)
    assert xn == pytest.approx(-1.0)
    assert extremum_from_fit(coeffs, 3, "max") == pytest.approx(1.0)
    assert extremum_from_fit(coeffs, 3, "min") == pytest.approx(-1.0)


def test_extrema_from_fit_returns_kinds():
    # y = x^2 -> one minimum at x=0
    quad = np.array([1.0, 0.0, 0.0])
    ex_q = extrema_from_fit(quad, 2)
    assert len(ex_q) == 1
    assert ex_q[0].kind == "min"
    assert ex_q[0].x == pytest.approx(0.0)

    # y = -x^3 + 3x -> max at +1, min at -1
    cubic = np.array([-1.0, 0.0, 3.0, 0.0])
    ex_c = extrema_from_fit(cubic, 3)
    assert {p.kind for p in ex_c} == {"max", "min"}
    by_kind = {p.kind: p.x for p in ex_c}
    assert by_kind["max"] == pytest.approx(1.0)
    assert by_kind["min"] == pytest.approx(-1.0)


def test_fit_polynomial_degrees():
    x = np.linspace(-1, 1, 10)
    y = x**2
    c2 = fit_polynomial(x, y, 2)
    assert c2[0] == pytest.approx(1.0, abs=0.05)
    c3 = fit_polynomial(x, y, 3)
    assert c3[0] == pytest.approx(0.0, abs=0.1)
