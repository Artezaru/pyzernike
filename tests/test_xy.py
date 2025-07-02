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
    results = xy_zernike_polynomial(X, Y, n=[2, 5, 4], m=[0, 1, 2], x_derivative=[0, 0, 0], y_derivative=[0, 1, 0], Rx=np.sqrt(2), Ry=np.sqrt(2), _skip=True)

    assert len(results) == 3, "Expected 3 results for n=[2, 5, 4] and m=[0, 1, 2]."
    assert all(result.shape == X.shape for result in results), "Result shapes do not match input shape."
    assert not np.any(np.isnan(results)), "Results contain NaN values."

    # Consistency check with flattened inputs
    flat_x = X.flatten()
    flat_y = Y.flatten()
    flat_results = xy_zernike_polynomial(flat_x, flat_y, n=[2, 5, 4], m=[0, 1, 2], x_derivative=[0, 0, 0], y_derivative=[0, 1, 0], Rx=np.sqrt(2), Ry=np.sqrt(2), _skip=True)

    assert len(flat_results) == 3, "Expected 3 results for flattened inputs."
    assert all(result.shape == (X.size,) for result in flat_results), "Flattened result shapes do not match input shape."
    assert not np.any(np.isnan(flat_results)), "Flattened results contain NaN values."

    assert all(np.allclose(results[i].flatten(), flat_results[i]) for i in range(3)), "Mismatch in flattened results for n=[2, 5, 4] and m=[0, 1, 2]."

