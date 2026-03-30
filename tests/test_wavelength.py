import pytest

from peakfit.wavelength import index_to_wavelength, linear_wavelengths


def test_index_to_wavelength_linear():
    w = linear_wavelengths(5, 400.0, 500.0)
    assert index_to_wavelength(w, 0.0) == pytest.approx(400.0)
    assert index_to_wavelength(w, 4.0) == pytest.approx(500.0)
    assert index_to_wavelength(w, 2.0) == pytest.approx(450.0)
    assert index_to_wavelength(w, 2.5) == pytest.approx(462.5)
