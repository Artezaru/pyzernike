import numpy as np
import pytest
from pyzernike import Z, Zxy


@pytest.mark.parametrize("n, m", [
    (4, 2),  # Test R_4^2
])
def test_extended_versus_circular(n, m):
    """Pytest function to validate global_radial_polynomial for R_4^2 and its derivative."""
    rho_bound = 10
    # Define test rho values
    rho_test = np.linspace(0, rho_bound - 1, 100)
    theta_test = np.linspace(0, 2 * np.pi, 100)
    rho_test, theta_test = np.meshgrid(rho_test, theta_test)

    # Compute using Zernike polynomial
    classic = Z(rho_test/rho_bound, theta_test, n, m)

    # Compute using extended Zernike polynomial
    x = rho_test * np.cos(theta_test)
    y = rho_test * np.sin(theta_test)

    extended = Zxy(x, y, n, m, Rx=rho_bound, Ry=rho_bound)

    # Assert that the results match within tolerance
    classic_nans = np.isnan(classic)
    extended_nans = np.isnan(extended)
    assert np.all(classic_nans == extended_nans), f"Mismatch in NaN values between classic and extended Zernike polynomials. NaNs in classic: {np.sum(classic_nans)}, NaNs in extended: {np.sum(extended_nans)}"
    classic = classic[~classic_nans]
    extended = extended[~extended_nans]
    assert np.allclose(classic, extended, atol=1e-8), (f"Mismatch in extended Zernike polynomial. Max difference: {np.abs(classic - extended).max()}")


