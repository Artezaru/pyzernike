import numpy as np
import pytest
from pyzernike import global_radial_polynomial

def explicit_R42(rho):
    """Explicit formula for R_4^2(rho): 4*rho^4 - 3*rho^2"""
    return 4 * np.power(rho, 4) - 3 * np.power(rho, 2)

def explicit_dR42(rho):
    """Explicit derivative for dR_4^2(rho): 16*rho^3 - 6*rho"""
    return 16 * np.power(rho, 3) - 6 * rho

@pytest.mark.parametrize("n, m, derivative, explicit_func", [
    (4, 2, 0, explicit_R42),        # Test R_4^2(rho)
    (4, 2, 1, explicit_dR42)        # Test d/d(rho) R_4^2(rho)
])
def test_global_radial_polynomial_R42(n, m, derivative, explicit_func):
    """Pytest function to validate global_radial_polynomial for R_4^2 and its derivative."""
    # Define test rho values
    rho_test = np.linspace(0, 1, 100)

    # Compute using global_radial_polynomial
    computed = global_radial_polynomial(rho_test, n, m, rho_derivative=derivative)

    # Compute explicit result
    expected = explicit_func(rho_test)

    # Assert that the results match within tolerance
    assert np.allclose(computed, expected, atol=1e-8), (
        f"Mismatch in R_{n}^{m} derivative order {derivative}.\n"
        f"Max difference: {np.abs(computed - expected).max()}"
    )
