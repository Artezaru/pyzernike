# Copyright 2025 Artezaru
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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


def test_zernike_polynomial_up_to_order_recursive():
    """Test that the zernike_polynomial_up_to_order function produces the same result as zernike_polynomial for the unit disk."""
    
    # Generate 100 random rho values between 0 and 1
    rho = np.linspace(0, 2.0, 100) # out of domain to test the behavior of the function
    theta = np.linspace(0, 2 * np.pi, 100)

    # All in one:
    common_result = zernike_polynomial_up_to_order(rho=rho, theta=theta, order=10, rho_derivative=[0,1,0], theta_derivative=[0,0,1], use_recursive=True)

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


def test_zernike_up_to_order_dimensions_recursive():
    """Test that the zernike_polynomial_up_to_order function returns results with the correct dimensions."""
    
    # Generate 100 random rho values between 0 and 1
    rho = np.linspace(0, 1, 10).astype(np.float64)
    theta = np.linspace(0, 2 * np.pi, 10).astype(np.float64)

    Rho, Theta = np.meshgrid(rho, theta, indexing='ij')

    # Test for different n and m values
    results = zernike_polynomial_up_to_order(rho=Rho, theta=Theta, order=10, rho_derivative=[0,1], theta_derivative=[0,0], use_recursive=True)

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


def test_xy_zernike_polynomial_up_to_order_recursive():
    """Test that the xy_zernike_polynomial_up_to_order function with use_recursive=True produces the same result as xy_zernike_polynomial for the unit disk."""

    # Generate 100 points on a diagonal, partly out of the domain
    x = np.linspace(-1, 1.2, 100) # out of domain to test the behavior of the function
    y = np.linspace(-1, 1.2, 100)

    # All in one:
    common_result = xy_zernike_polynomial_up_to_order(x=x, y=y, order=10, x_derivative=[0,1,0], y_derivative=[0,0,1], Rx=np.sqrt(2), Ry=np.sqrt(2), use_recursive=True)

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


@pytest.mark.parametrize("h", [0.0, 0.3])
def test_xy_zernike_polynomial_up_to_order_recursive_annulus(h):
    """Test that the recursive and the explicit methods of xy_zernike_polynomial_up_to_order give the same results on a rotated elliptic annulus, for derivatives up to the order 3."""

    rng = np.random.default_rng(0)
    x, y = rng.uniform(-1.6, 1.6, (2, 2000))
    x[0], y[0] = 0.2, -0.1  # exact center of the ellipse
    ellipse = dict(Rx=1.6, Ry=1.1, x0=0.2, y0=-0.1, alpha=0.5, h=h)
    derivatives = dict(x_derivative=[0, 1, 0, 2, 1, 0, 3], y_derivative=[0, 0, 1, 0, 1, 2, 1])

    recursive_result = xy_zernike_polynomial_up_to_order(x=x, y=y, order=10, **derivatives, **ellipse, use_recursive=True, to_list=False)
    explicit_result = xy_zernike_polynomial_up_to_order(x=x, y=y, order=10, **derivatives, **ellipse, use_recursive=False, to_list=False)

    assert np.array_equal(np.isnan(recursive_result), np.isnan(explicit_result)), f"The domains of the two methods differ for h={h}."
    scale = np.maximum(1, np.nanmax(np.abs(explicit_result), axis=-1, keepdims=True))
    assert np.nanmax(np.abs(recursive_result - explicit_result) / scale) < 1e-10, f"Mismatch between the recursive and the explicit methods for h={h}."


def test_xy_zernike_polynomial_up_to_order_recursive_center():
    """Test that the xy_zernike_polynomial_up_to_order function with use_recursive=True is finite at the center of a full ellipse."""

    x = np.array([0.3])
    y = np.array([-0.2])
    result = xy_zernike_polynomial_up_to_order(x=x, y=y, order=15, x_derivative=[0, 1, 0, 2, 1, 0], y_derivative=[0, 0, 1, 0, 1, 2], Rx=1.6, Ry=1.1, x0=0.3, y0=-0.2, alpha=0.5, use_recursive=True, to_list=False)

    assert np.all(np.isfinite(result)), "Non finite values at the center of the ellipse."


def test_xy_zernike_up_to_order_dimensions_recursive():
    """Test that the xy_zernike_polynomial_up_to_order function with use_recursive=True returns results with the correct dimensions."""

    x = np.linspace(-1, 1, 10).astype(np.float64)
    y = np.linspace(-1, 1, 10).astype(np.float64)

    X, Y = np.meshgrid(x, y, indexing='ij')

    # Test for different n and m values
    results = xy_zernike_polynomial_up_to_order(x=X, y=Y, order=10, x_derivative=[0,1], y_derivative=[0,0], Rx=np.sqrt(2), Ry=np.sqrt(2), use_recursive=True)

    assert len(results) == 2 and all(len(result) == 66 for result in results), "Expected 2 lists of 66 arrays."
    assert all(array.shape == X.shape for result in results for array in result), "Result shapes do not match input shape."

    # Consistency check with flattened inputs
    flat_x = X.flatten()
    flat_y = Y.flatten()
    flat_results = xy_zernike_polynomial_up_to_order(flat_x, flat_y, order=10, x_derivative=[0,1], y_derivative=[0,0], Rx=np.sqrt(2), Ry=np.sqrt(2), use_recursive=True)

    assert all(np.allclose(results[i][j].flatten(), flat_results[i][j]) for i in range(len(results)) for j in range(len(results[i]))), (
        "Mismatch between results and flat_results."
    )


def test_xy_zernike_up_to_order_dtype_recursive():
    """Test that the xy_zernike_polynomial_up_to_order function with use_recursive=True returns results of the correct dtype."""
    x_32 = np.linspace(-1, 1, 100).astype(np.float32)
    y_32 = np.linspace(-1, 1, 100).astype(np.float32)
    x_64 = np.linspace(-1, 1, 100).astype(np.float64)
    y_64 = np.linspace(-1, 1, 100).astype(np.float64)

    for x, y, dtype in [
        (x_32, y_32, np.float32),
        (x_64, y_64, np.float64),
    ]:
        for h in [0.0, 0.3]:
            result = xy_zernike_polynomial_up_to_order(x=x, y=y, order=10, x_derivative=[0, 1], y_derivative=[0, 0], Rx=np.sqrt(2), Ry=np.sqrt(2), h=h, use_recursive=True)
            for derivative_array in result:
                for poly_array in derivative_array:
                    assert poly_array.dtype == dtype, f"Expected dtype {dtype}, but got {poly_array.dtype} for h={h}"