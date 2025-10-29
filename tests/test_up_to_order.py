import pytest
import numpy as np
from pyzernike import zernike_polynomial_up_to_order, zernike_polynomial, zernike_order_to_index, xy_zernike_polynomial_up_to_order, xy_zernike_polynomial

def test_zernike_polynomial_up_to_order():
    """Test that the zernike_polynomial_up_to_order function produces the same result as zernike_polynomial for the unit disk."""
    
    # Generate 100 random rho values between 0 and 1
    rho = np.linspace(0, 2.0, 100) # out of domain to test the behavior of the function
    theta = np.linspace(0, 2 * np.pi, 100)

    # All in one:
    common_result = zernike_polynomial_up_to_order(rho=rho, theta=theta, order=10, rho_derivative=[0,1,0], theta_derivative=[0,0,1])

    assert len(common_result) == 3, "The output should contain three arrays: one for rho_derivative, one for theta_derivative, and one for the Zernike polynomials."
    assert len(common_result[0]) == 66, f"The first output array should contain 66 elements, corresponding to the Zernike polynomials up to order 10, but got {len(common_result[0])}."

    for n in range(10):
        for m in range(-n, n + 1, 2):
            zernike_result = zernike_polynomial(rho=rho, theta=theta, n=[n,n,n], m=[m,m,m], rho_derivative=[0, 1, 0], theta_derivative=[0, 0, 1])
            index = zernike_order_to_index([n], [m])[0]

            # Check if the result matches the common result
            assert np.allclose(common_result[0][index], zernike_result[0], equal_nan=True), (
                f"Mismatch for n={n}, m={m}. Expected: {zernike_result[0]}, Got: {common_result[0][index]}"
            )
            assert np.allclose(common_result[1][index], zernike_result[1], equal_nan=True), (
                f"Mismatch for n={n}, m={m}. Expected: {zernike_result[1]}, Got: {common_result[1][index]}"
            )
            assert np.allclose(common_result[2][index], zernike_result[2], equal_nan=True), (
                f"Mismatch for n={n}, m={m}. Expected: {zernike_result[2]}, Got: {common_result[2][index]}"
            )


def test_zernike_up_to_order_dimensions():
    """Test that the zernike_polynomial_up_to_order function returns results with the correct dimensions."""
    
    # Generate 100 random rho values between 0 and 1
    rho = np.linspace(0, 1, 10).astype(np.float64)
    theta = np.linspace(0, 2 * np.pi, 10).astype(np.float64)

    Rho, Theta = np.meshgrid(rho, theta, indexing='ij')

    # Test for different n and m values
    results = zernike_polynomial_up_to_order(rho=Rho, theta=Theta, order=10, rho_derivative=[0,1], theta_derivative=[0,0])

    # Consistency check with flattened inputs
    flat_rho = Rho.flatten()
    flat_theta = Theta.flatten()
    flat_results = zernike_polynomial_up_to_order(flat_rho, flat_theta, order=10, rho_derivative=[0,1], theta_derivative=[0,0])

    assert all(np.allclose(results[i][j].flatten(), flat_results[i][j]) for i in range(len(results)) for j in range(len(results[i]))), (
        "Mismatch between results and flat_results."
    )


def test_xy_zernike_polynomial_up_to_order():
    """Test that the xy_zernike_polynomial_up_to_order function produces the same result as xy_zernike_polynomial for the unit disk."""
    
    # Generate 100 random rho values between 0 and 1
    x = np.linspace(-1, 1.2, 100) # out of domain to test the behavior of the function
    y = np.linspace(-1, 1.2, 100)

    # All in one:
    common_result = xy_zernike_polynomial_up_to_order(x=x, y=y, order=10, x_derivative=[0,1,0], y_derivative=[0,0,1], Rx=np.sqrt(2), Ry=np.sqrt(2))

    assert len(common_result) == 3, "The output should contain three arrays: one for x_derivative, one for y_derivative, and one for the Zernike polynomials."
    assert len(common_result[0]) == 66, f"The first output array should contain 66 elements, corresponding to the Zernike polynomials up to order 10, but got {len(common_result[0])}."

    for n in range(10):
        for m in range(-n, n + 1, 2):
            xy_result = xy_zernike_polynomial(x=x, y=y, n=[n,n,n], m=[m,m,m], x_derivative=[0, 1, 0], y_derivative=[0, 0, 1], Rx=np.sqrt(2), Ry=np.sqrt(2))
            index = zernike_order_to_index([n], [m])[0]

            # Check if the result matches the common result
            assert np.allclose(common_result[0][index], xy_result[0], equal_nan=True), (
                f"Mismatch for n={n}, m={m}. Expected: {xy_result[0]}, Got: {common_result[0][index]}"
            )
            assert np.allclose(common_result[1][index], xy_result[1], equal_nan=True), (
                f"Mismatch for n={n}, m={m}. Expected: {xy_result[1]}, Got: {common_result[1][index]}"
            )
            assert np.allclose(common_result[2][index], xy_result[2], equal_nan=True), (
                f"Mismatch for n={n}, m={m}. Expected: {xy_result[2]}, Got: {common_result[2][index]}"
            )


def test_xy_zernike_up_to_order_dimensions():
    """Test that the xy_zernike_polynomial_up_to_order function returns results with the correct dimensions."""
    
    # Generate 100 random rho values between 0 and 1
    x = np.linspace(-1, 1, 10).astype(np.float64)
    y = np.linspace(-1, 1, 10).astype(np.float64)

    X, Y = np.meshgrid(x, y, indexing='ij')

    # Test for different n and m values
    results = xy_zernike_polynomial_up_to_order(x=X, y=Y, order=10, x_derivative=[0,1], y_derivative=[0,0], Rx=np.sqrt(2), Ry=np.sqrt(2))

    # Consistency check with flattened inputs
    flat_x = X.flatten()
    flat_y = Y.flatten()
    flat_results = xy_zernike_polynomial_up_to_order(flat_x, flat_y, order=10, x_derivative=[0,1], y_derivative=[0,0], Rx=np.sqrt(2), Ry=np.sqrt(2))

    assert all(np.allclose(results[i][j].flatten(), flat_results[i][j]) for i in range(len(results)) for j in range(len(results[i]))), (
        "Mismatch between results and flat_results."
    )
    

def test_zernike_up_to_order_dtype():
    """Test that the zernike_polynomial function returns results of the correct dtype."""
    rho_16 = np.linspace(0, 1, 100).astype(np.float16)
    theta_16 = np.linspace(0, 2 * np.pi, 100).astype(np.float16)
    rho_32 = np.linspace(0, 1, 100).astype(np.float32)
    theta_32 = np.linspace(0, 2 * np.pi, 100).astype(np.float32)
    rho_64 = np.linspace(0, 1, 100).astype(np.float64)
    theta_64 = np.linspace(0, 2 * np.pi, 100).astype(np.float64)

    for rho, theta, dtype in [
        (rho_16, theta_16, np.float16),
        (rho_32, theta_32, np.float32),
        (rho_64, theta_64, np.float64),
    ]:
        result = zernike_polynomial_up_to_order(rho=rho, theta=theta, order=10, rho_derivative=[0], theta_derivative=[0])
        for derivative_array in result:
            for poly_array in derivative_array:
                assert poly_array.dtype == dtype, f"Expected dtype {dtype}, but got {poly_array.dtype}"


