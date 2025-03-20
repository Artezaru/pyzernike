import numpy as np
import pytest
from pyzernike import global_radial_polynomial, litteral_radial_polynomial

# Generate 10 sets of random (n, m, derivative) combinations for the tests
test_cases = []
for _ in range(10):
    n = np.random.randint(0, 6)
    m_valid = [k for k in range(0, n + 1) if (n - k) % 2 == 0]
    m = np.random.choice(m_valid)
    derivative = np.random.randint(0, n+1)
    test_cases.append((n, m, derivative))

@pytest.mark.parametrize("n, m, derivative", test_cases)
def test_litteral_vs_global_radial_polynomial_multiple(n, m, derivative):
    """Compare global_radial_polynomial and litteral_radial_polynomial for random (n, m, derivative)."""

    # Generate 10 random rho values between 0 and 1
    rho_test = np.random.rand(10)

    # Compute using global_radial_polynomial
    global_result = global_radial_polynomial(rho_test, n, m, rho_derivative=derivative)

    # Compute using symbolic_radial_polynomial
    litteral_result = litteral_radial_polynomial(rho_test, n, m, rho_derivative=derivative)

    # Assert that the results match within tolerance
    assert np.allclose(global_result, litteral_result, atol=1e-8), (
        f"Mismatch for n={n}, m={m}, derivative={derivative}.\n"
        f"Max difference: {np.abs(global_result - litteral_result).max()}"
    )
