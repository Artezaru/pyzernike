import pytest
import numpy as np
from pyzernike import xy_zernike_polynomial, zernike_polynomial


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


def test_xy_zernike_polynomial_derivatives():
    """Test that the xy_zernike_polynomial function produces same result as zernike_polynomial for the unit disk."""
    
    # Generate 100 random rho values between 0 and 1
    radius = 10
    rho = np.linspace(0.1, 1.0, 100) # avoid zero to prevent division by zero
    theta = np.linspace(0, 2 * np.pi, 100)

    for n in range(15):
        for m in range(-n, n + 1, 2):
            zernike_result = zernike_polynomial(rho=rho, theta=theta, n=[n,n], m=[m,m], rho_derivative=[1,0], theta_derivative=[0,1])

            x = radius * rho * np.cos(theta)
            y = radius * rho * np.sin(theta)

            rho_bis = np.sqrt(x**2 + y**2) / radius
            theta_bis = np.arctan2(y/radius, x/radius)
            theta_bis = np.mod(theta_bis, 2 * np.pi)

            assert np.allclose(rho_bis, rho, equal_nan=True), (
                f"Mismatch in rho values for n={n}, m={m}. Expected: {rho_bis}, Got: {rho}"
            )
            assert np.allclose(theta_bis, theta, equal_nan=True), (
                f"Mismatch in theta values for n={n}, m={m}. Expected: {theta_bis}, Got: {theta}"
            )

            rho_dx = x / (radius * np.sqrt(x**2 + y**2))
            rho_dy = y / (radius * np.sqrt(x**2 + y**2))
            theta_dx = -y / (x**2 + y**2)
            theta_dy = x / (x**2 + y**2)

            zernike_result_dx = zernike_result[0] * rho_dx + zernike_result[1] * theta_dx
            zernike_result_dy = zernike_result[0] * rho_dy + zernike_result[1] * theta_dy

            xy_result = xy_zernike_polynomial(x=x, y=y, n=[n,n], m=[m,m], x_derivative=[1,0], y_derivative=[0,1], Rx=radius, Ry=radius)

            assert np.allclose(zernike_result_dx, xy_result[0], equal_nan=True), (
                f"Mismatch in x derivative for n={n}, m={m}. Expected: {zernike_result_dx}, Got: {xy_result[0]}"
            )
            assert np.allclose(zernike_result_dy, xy_result[1], equal_nan=True), (
                f"Mismatch in y derivative for n={n}, m={m}. Expected: {zernike_result_dy}, Got: {xy_result[1]}"
            )


def test_xy_zernike_dimensions():
    """Test that the zernike_polynomial function returns results with the correct dimensions."""
    
    # Generate 100 random rho values between 0 and 1
    x = np.linspace(-1,1,100)
    y = np.linspace(-1,1,100)

    X, Y = np.meshgrid(x, y, indexing='ij')

    # Test for different n and m values
    results = xy_zernike_polynomial(X, Y, n=[2, 5, 4], m=[0, 1, 2], x_derivative=[0, 0, 0], y_derivative=[0, 1, 0], Rx=np.sqrt(2), Ry=np.sqrt(2))

    assert len(results) == 3, "Expected 3 results for n=[2, 5, 4] and m=[0, 1, 2]."
    assert all(result.shape == X.shape for result in results), "Result shapes do not match input shape."
    assert not np.any(np.isnan(results)), "Results contain NaN values."

    # Consistency check with flattened inputs
    flat_x = X.flatten()
    flat_y = Y.flatten()
    flat_results = xy_zernike_polynomial(flat_x, flat_y, n=[2, 5, 4], m=[0, 1, 2], x_derivative=[0, 0, 0], y_derivative=[0, 1, 0], Rx=np.sqrt(2), Ry=np.sqrt(2))

    assert len(flat_results) == 3, "Expected 3 results for flattened inputs."
    assert all(result.shape == (X.size,) for result in flat_results), "Flattened result shapes do not match input shape."
    assert not np.any(np.isnan(flat_results)), "Flattened results contain NaN values."

    assert all(np.allclose(results[i].flatten(), flat_results[i]) for i in range(3)), "Mismatch in flattened results for n=[2, 5, 4] and m=[0, 1, 2]."



def test_default_value():
    """ Test that the xy_zernike_polynomial function correctly assigns the default value for out-of-domain rho values."""
    lx = np.linspace(-2, 2, 100)
    ly = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(lx, ly, indexing='ij')

    Rho = np.sqrt(X**2 + Y**2)

    result = xy_zernike_polynomial(X, Y, n=[2], m=[0], x_derivative=[0], y_derivative=[0], default=-10.0)[0]
    assert np.all(result[Rho > 1] == -10.0), "Default value not correctly assigned for out-of-domain rho values."


def test_optional_derivatives():
    """ Test if rho and theta derivatives are set to 0 by default in xy_zernike_polynomial function."""
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    
    for x_derivative in [None, [0,1,0]]:
        for y_derivative in [None, [0,0,1]]:

            result = xy_zernike_polynomial(x=x, y=y, n=[3, 2, 1], m=[1, 0, -1], x_derivative=x_derivative, y_derivative=y_derivative)

            result_explicit = xy_zernike_polynomial(x=x, y=y, n=[3, 2, 1], m=[1, 0, -1], x_derivative=x_derivative if x_derivative is not None else ([0]*len(y_derivative) if y_derivative is not None else [0, 0, 0]), y_derivative=y_derivative if y_derivative is not None else ([0]*len(x_derivative) if x_derivative is not None else [0, 0, 0]))

            for i in range(len(result)):
                assert np.allclose(result[i], result_explicit[i], equal_nan=True), (
                    f"Mismatch for x_derivative={x_derivative}, y_derivative={y_derivative}. "
                    f"Expected: {result_explicit[i]}, Got: {result[i]}"
                )

def test_zernike_dtype():
    """Test that the xy_zernike_polynomial function returns results of the correct dtype."""
    x_16 = np.linspace(0, 1, 100).astype(np.float16)
    y_16 = np.linspace(0, 1, 100).astype(np.float16)
    x_32 = np.linspace(0, 1, 100).astype(np.float32)
    y_32 = np.linspace(0, 1, 100).astype(np.float32)
    x_64 = np.linspace(0, 1, 100).astype(np.float64)
    y_64 = np.linspace(0, 1, 100).astype(np.float64)

    for n in range(5):
        for m in range(-n, n + 1, 2):
            result_16 = xy_zernike_polynomial(x=x_16, y=y_16, n=[n], m=[m], x_derivative=[0], y_derivative=[0])[0]
            result_32 = xy_zernike_polynomial(x=x_32, y=y_32, n=[n], m=[m], x_derivative=[0], y_derivative=[0])[0]
            result_64 = xy_zernike_polynomial(x=x_64, y=y_64, n=[n], m=[m], x_derivative=[0], y_derivative=[0])[0]

            assert result_16.dtype == np.float16, f"Expected dtype float16, got {result_16.dtype} for n={n}, m={m}"
            assert result_32.dtype == np.float32, f"Expected dtype float32, got {result_32.dtype} for n={n}, m={m}"
            assert result_64.dtype == np.float64, f"Expected dtype float64, got {result_64.dtype} for n={n}, m={m}"
