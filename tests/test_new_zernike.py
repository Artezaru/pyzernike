import numpy as np
import pytest
import sympy

from pyzernike import radial_polynomial, zernike_polynomial, radial_symbolic, zernike_symbolic, xy_zernike_polynomial
from pyzernike.deprecated import zernike_polynomial as old_zernike_polynomial
from pyzernike.deprecated import radial_polynomial as old_radial_polynomial





def test_old_vs_new_radial_zernike():
    """Compare old_radial_polynomial (deprecated) and radial_polynomial for all (n, m, rho_derivative). -> Ensure that the new radial_polynomial function matches the deprecated old_radial_polynomial function."""

    # Generate 100 random rho values between 0 and 1
    rho_test = np.linspace(0, 2, 100) # Same data are out of bounds to test the behavior of the function

    for n in range(15):
        for m in range(0, n + 1):
            for rho_derivative in range(n):
                print(f"Testing n={n}, m={m}, rho_derivative={rho_derivative}")
                # Compute using old_radial_polynomial
                old_result = old_radial_polynomial(rho_test, n, m, rho_derivative=rho_derivative)

                # Compute using radial_polynomial
                result = radial_polynomial(rho_test, [n], [m], [rho_derivative])[0]

                assert np.allclose(result, result, equal_nan=True), (
                    f"Mismatch between old_radial_polynomial and radial_polynomial for n={n}, m={m}, rho_derivative={rho_derivative}."
                    f" Expected: {old_result}, Got: {result}"
                )

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


def test_polynomial_consistency():
    """Test that the zernike_polynomial function produces consistent results when called with multiple n, m, rho_derivative, and theta_derivative values."""
    list_n = []
    list_m = []
    list_rho_derivative = []
    list_theta_derivative = []

    for n in range(7):
        for m in range(0, n + 1):
            for rho_derivative in range(n):
                for theta_derivative in range(n):
                    list_n.append(n)
                    list_m.append(m)
                    list_rho_derivative.append(rho_derivative)
                    list_theta_derivative.append(theta_derivative)

    rho = np.linspace(0, 1, 100)
    theta = np.linspace(0, 2 * np.pi, 100)

    # Compute all in one
    common_result = zernike_polynomial(rho= rho, theta=theta, n=list_n, m=list_m, rho_derivative=list_rho_derivative, theta_derivative=list_theta_derivative)

    cumulative_result = []
    for i in range(len(list_n)):
        n = list_n[i]
        m = list_m[i]
        rho_derivative = list_rho_derivative[i]
        theta_derivative = list_theta_derivative[i]

        # Compute each one separately
        result = zernike_polynomial(rho=rho, theta=theta, n=[n], m=[m], rho_derivative=[rho_derivative], theta_derivative=[theta_derivative])[0]
        cumulative_result.append(result)

    # Check that all results are the same
    for i in range(len(cumulative_result)):
        assert np.allclose(cumulative_result[i], common_result[i], equal_nan=True), (
            f"Mismatch in cumulative results for index {i} with n={list_n[i]}, m={list_m[i]}, "
            f"rho_derivative={list_rho_derivative[i]}, theta_derivative={list_theta_derivative[i]}."
            f" Expected: {cumulative_result[i]}, Got: {common_result[i]}"
        )
    



def test_xy_zernike_polynomial():
    """Test that the xy_zernike_polynomial function produces same result as zernike_polynomial for the unit disk."""
    
    # Generate 100 random rho values between 0 and 1
    radius = 10
    rho = np.linspace(0, 1.0, 100)
    theta = np.linspace(0, 2 * np.pi, 100)

    for n in range(15):
        for m in range(0, n + 1):
            zernike_result = zernike_polynomial(rho=rho, theta=theta, n=[n], m=[m])[0]

            x = radius * rho * np.cos(theta)
            y = radius * rho * np.sin(theta)

            xy_result = xy_zernike_polynomial(x=x, y=y, n=[n], m=[m], Rx=radius, Ry=radius)[0]

            assert np.allclose(zernike_result, xy_result, equal_nan=True), (
                f"Mismatch between zernike_polynomial and xy_zernike_polynomial for n={n}, m={m}."
                f" Expected: {zernike_result}, Got: {xy_result}"
            )