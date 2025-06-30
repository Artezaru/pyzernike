import numpy as np
import pytest
from pyzernike.deprecated import global_radial_polynomial, litteral_radial_polynomial, symbolic_radial_polynomial, radial_polynomial, Z, Zxy

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

# Generate 10 sets of random (n, m, derivative) combinations for the tests
litteral_test_cases = []
for _ in range(10):
    n = np.random.randint(0, 6)
    m_valid = [k for k in range(0, n + 1) if (n - k) % 2 == 0]
    m = np.random.choice(m_valid)
    derivative = np.random.randint(0, n+1)
    litteral_test_cases.append((n, m, derivative))

@pytest.mark.parametrize("n, m, derivative", litteral_test_cases)
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

def test_litteral_radial_polynomial_higher():
    """Test the litteral_radial_polynomial for higher n and m values."""
    
    # Test with n = 10, m = 0, rho_derivative = 0
    rho_test = np.random.rand(10)
    result = litteral_radial_polynomial(rho_test, 10, 0, rho_derivative=0)
    
    # Check if the result is of the expected shape
    assert result.shape == rho_test.shape, "Result shape mismatch for n=10, m=0."

    # Test with n = 10, m = 2, rho_derivative = 1
    result = litteral_radial_polynomial(rho_test, 10, 2, rho_derivative=1)
    
    # Check if the result is of the expected shape
    assert result.shape == rho_test.shape, "Result shape mismatch for n=10, m=2."

# Generate 10 sets of random (n, m, derivative) combinations for the tests
symbolic_test_cases = []
for _ in range(10):
    n = np.random.randint(0, 11)
    m_valid = [k for k in range(0, n + 1) if (n - k) % 2 == 0]
    m = np.random.choice(m_valid)
    derivative = np.random.randint(0, n+1)
    symbolic_test_cases.append((n, m, derivative))

@pytest.mark.parametrize("n, m, derivative", symbolic_test_cases)
def test_symbolic_vs_global_radial_polynomial_multiple(n, m, derivative):
    """Compare global_radial_polynomial and symbolic_radial_polynomial for random (n, m, derivative)."""

    # Generate 10 random rho values between 0 and 1
    rho_test = np.random.rand(10)

    # Compute using global_radial_polynomial
    global_result = global_radial_polynomial(rho_test, n, m, rho_derivative=derivative)

    # Compute using symbolic_radial_polynomial
    litteral_result = symbolic_radial_polynomial(rho_test, n, m, rho_derivative=derivative)

    # Assert that the results match within tolerance
    assert np.allclose(global_result, litteral_result, atol=1e-8), (
        f"Mismatch for n={n}, m={m}, derivative={derivative}.\n"
        f"Max difference: {np.abs(global_result - litteral_result).max()}"
    )

# Generate 10 sets of random (n, m, derivative) combinations for the tests
test_cases = []
for _ in range(10):
    n = np.random.randint(0, 6)
    m_valid = [k for k in range(0, n + 1) if (n - k) % 2 == 0]
    m = np.random.choice(m_valid)
    derivative = np.random.randint(0, n+1)
    test_cases.append((n, m, derivative))

@pytest.mark.parametrize("n, m, derivative", test_cases)
def test_vs_global_radial_polynomial_multiple(n, m, derivative):
    """Compare global_radial_polynomial and radial_polynomial for random (n, m, derivative)."""

    # Generate 10 random rho values between 0 and 1
    rho_test = np.random.rand(10)

    # Compute using global_radial_polynomial
    global_result = global_radial_polynomial(rho_test, n, m, rho_derivative=derivative)

    # Compute using symbolic_radial_polynomial
    result = radial_polynomial(rho_test, n, m, rho_derivative=derivative)

    # Assert that the results match within tolerance
    assert np.allclose(global_result, result, atol=1e-8), (
        f"Mismatch for n={n}, m={m}, derivative={derivative}.\n"
        f"Max difference: {np.abs(global_result - result).max()}"
    )


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


