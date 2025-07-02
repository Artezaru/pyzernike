import numpy as np
import pytest
import sympy

from pyzernike import radial_polynomial, zernike_polynomial, radial_symbolic, zernike_symbolic



def test_symbolic_radial():
    """Test that the symbolic radial polynomial matches the computed radial polynomial for a range of (n, m, rho_derivative) values."""
    
    # Generate 100 random rho values between 0 and 1
    rho_test = np.linspace(0, 1, 100)

    for n in range(15):
        for m in range(0, n + 1):
            for rho_derivative in range(n):
                print(f"Testing n={n}, m={m}, rho_derivative={rho_derivative}")
                # Compute using symbolic radial polynomial
                symbolic_expression = radial_symbolic([n], [m], [rho_derivative])[0]

                # `x` represents the radial coordinate in the symbolic expression
                func = sympy.lambdify('x', symbolic_expression, 'numpy')
                symbolic_result = func(rho_test)

                # Compute using core_polynomial
                result = radial_polynomial(rho=rho_test, n=[n], m=[m], rho_derivative=[rho_derivative])[0]

                assert np.allclose(symbolic_result, result), (
                    f"Mismatch between symbolic and computed radial polynomial for n={n}, m={m}, rho_derivative={rho_derivative}."
                    f" Expected: {symbolic_result}, Got: {result}"
                )


def test_symbolic_zernike():
    """Test that the symbolic polynomial matches the computed zernike polynomial for a range of (n, m, rho_derivative, theta_derivative) values."""
    
    # Generate 100 random rho values between 0 and 1
    rho = np.linspace(0, 1, 100)
    theta = np.linspace(0, 2 * np.pi, 100)

    for n in range(15):
        for m in range(0, n + 1):
            for rho_derivative in range(n):
                for theta_derivative in range(n):
                    print(f"Testing n={n}, m={m}, rho_derivative={rho_derivative}, theta_derivative={theta_derivative}")
                    # Compute using symbolic zernike polynomial
                    symbolic_expression = zernike_symbolic([n], [m], [rho_derivative], [theta_derivative])[0]

                    # `x` represents the radial coordinate in the symbolic expression
                    # `y` represents the angular coordinate in the symbolic expression
                    func = sympy.lambdify(['x', 'y'], symbolic_expression, 'numpy')
                    symbolic_result = func(rho, theta)

                    # Compute using core_polynomial
                    result = zernike_polynomial(rho=rho, theta=theta, n=[n], m=[m], rho_derivative=[rho_derivative], theta_derivative=[theta_derivative])[0]

                    assert np.allclose(symbolic_result, result), (
                        f"Mismatch between symbolic and computed zernike polynomial for n={n}, m={m}, "
                        f"rho_derivative={rho_derivative}, theta_derivative={theta_derivative}."
                        f" Expected: {symbolic_result}, Got: {result}"
                    )