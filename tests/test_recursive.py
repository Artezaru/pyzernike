import math
from fractions import Fraction

import numpy as np
import pytest

try:
    import pyzernike
except:
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    import pyzernike

from pyzernike.core.core_polynomial import _core_polynomials
from pyzernike.core.zernike_recursive import _zernike_recursive_up_to_order
from pyzernike.tools import zernike_index_to_order, zernike_polynomials_count


def _polar_grid(n_rho=30, n_theta=40, dtype=np.float64):
    """Polar grid covering the whole unit disk, including rho = 0 and rho = 1."""
    rho = np.linspace(0, 1, n_rho, dtype=dtype)
    theta = np.linspace(0, 2 * np.pi, n_theta, dtype=dtype)
    return np.meshgrid(rho, theta, indexing='ij')


def _core_up_to_order(rho, theta, order, rho_derivatives, theta_derivatives):
    """Compute all the polynomials up to `order` with `_core_polynomials`, with the output layout (D, P, ...) of the recursion."""
    n_polynomials = zernike_polynomials_count(order)
    n, m = zernike_index_to_order(list(range(n_polynomials)))
    return np.stack([
        _core_polynomials(rho, theta, n, m, [dr] * n_polynomials, [dt] * n_polynomials, False, True)
        for dr, dt in zip(rho_derivatives, theta_derivatives)
    ])


def _assert_close(recursive, reference, message, rtol=1e-10):
    """Compare two sets of polynomials with a tolerance relative to the magnitude of each polynomial."""
    scale = np.maximum(1.0, np.max(np.abs(reference), axis=tuple(range(1, reference.ndim)), keepdims=True))
    error = np.max(np.abs(recursive - reference) / scale)
    assert error < rtol, f"{message} Max relative error: {error:.2e}"


def test_recursive_matches_core_values():
    """Test that the recursion gives the same polynomials as the explicit sum for all the orders up to 12."""
    Rho, Theta = _polar_grid()

    for order in range(13):
        recursive = _zernike_recursive_up_to_order(Rho, Theta, order, [0], [0])
        reference = _core_up_to_order(Rho, Theta, order, [0], [0])

        assert recursive.shape == reference.shape, f"Shape mismatch for order={order}: {recursive.shape} vs {reference.shape}."
        _assert_close(recursive, reference, f"Mismatch between recursion and explicit sum for order={order}.")


def test_recursive_matches_core_derivatives():
    """Test that the recursion gives the same derivatives as the explicit sum for all the pairs (rho_derivative, theta_derivative) up to 4."""
    Rho, Theta = _polar_grid()
    order = 10

    rho_derivatives = [dr for dr in range(5) for _ in range(5)]
    theta_derivatives = [dt for _ in range(5) for dt in range(5)]

    recursive = _zernike_recursive_up_to_order(Rho, Theta, order, rho_derivatives, theta_derivatives)
    reference = _core_up_to_order(Rho, Theta, order, rho_derivatives, theta_derivatives)

    for d, (dr, dt) in enumerate(zip(rho_derivatives, theta_derivatives)):
        _assert_close(recursive[d], reference[d], f"Mismatch for rho_derivative={dr}, theta_derivative={dt}.")


def test_recursive_derivative_pairs_consistency():
    """Test that computing several derivative pairs at once gives the same results as computing them one by one."""
    Rho, Theta = _polar_grid()
    order = 8
    rho_derivatives = [0, 1, 0, 2, 1, 3]
    theta_derivatives = [0, 0, 1, 0, 1, 2]

    common_result = _zernike_recursive_up_to_order(Rho, Theta, order, rho_derivatives, theta_derivatives)

    for d, (dr, dt) in enumerate(zip(rho_derivatives, theta_derivatives)):
        single_result = _zernike_recursive_up_to_order(Rho, Theta, order, [dr], [dt])[0]
        assert np.array_equal(common_result[d], single_result), (
            f"Mismatch between common and single computation for rho_derivative={dr}, theta_derivative={dt}."
        )


def test_recursive_known_polynomials():
    """Test the recursion against the closed-form expressions of the first polynomials and of some derivatives."""
    Rho, Theta = _polar_grid()
    Z, dZ_drho, dZ_dtheta = _zernike_recursive_up_to_order(Rho, Theta, 2, [0, 1, 0], [0, 0, 1])

    # OSA/ANSI ordering: j = 0: Z_0^0, 1: Z_1^-1, 2: Z_1^1, 3: Z_2^-2, 4: Z_2^0, 5: Z_2^2
    expected = {
        0: (np.ones_like(Rho), np.zeros_like(Rho), np.zeros_like(Rho)),
        1: (Rho * np.sin(Theta), np.sin(Theta), Rho * np.cos(Theta)),
        2: (Rho * np.cos(Theta), np.cos(Theta), -Rho * np.sin(Theta)),
        3: (Rho**2 * np.sin(2 * Theta), 2 * Rho * np.sin(2 * Theta), 2 * Rho**2 * np.cos(2 * Theta)),
        4: (2 * Rho**2 - 1, 4 * Rho, np.zeros_like(Rho)),
        5: (Rho**2 * np.cos(2 * Theta), 2 * Rho * np.cos(2 * Theta), -2 * Rho**2 * np.sin(2 * Theta)),
    }

    for j, (value, derivative_rho, derivative_theta) in expected.items():
        assert np.allclose(Z[j], value), f"Mismatch for the polynomial j={j}."
        assert np.allclose(dZ_drho[j], derivative_rho), f"Mismatch for the rho derivative of the polynomial j={j}."
        assert np.allclose(dZ_dtheta[j], derivative_theta), f"Mismatch for the theta derivative of the polynomial j={j}."


def test_recursive_derivative_higher_than_order():
    """Test that the rho derivatives of order greater than n are exactly zero."""
    Rho, Theta = _polar_grid()
    order = 6
    n, _ = zernike_index_to_order(list(range(zernike_polynomials_count(order))))

    for dr in range(1, order + 3):
        result = _zernike_recursive_up_to_order(Rho, Theta, order, [dr], [0])[0]
        for j, n_j in enumerate(n):
            if dr > n_j:
                assert np.all(result[j] == 0.0), f"Non-zero derivative of order {dr} for the polynomial j={j} (n={n_j})."


def test_recursive_unit_circle():
    """Test that R_n^m(1) = 1 for all the orders up to 30, i.e. Z_n^m(1, 0) = 1 for m >= 0."""
    order = 30
    n, m = zernike_index_to_order(list(range(zernike_polynomials_count(order))))

    result = _zernike_recursive_up_to_order(np.array([1.0]), np.array([0.0]), order, [0], [0])[0, :, 0]
    for j, (n_j, m_j) in enumerate(zip(n, m)):
        if m_j >= 0:
            assert np.isclose(result[j], 1.0), f"R_{n_j}^{m_j}(1) = {result[j]} instead of 1."


def test_recursive_high_order_accuracy():
    """Test that the recursion stays accurate at high orders, where the explicit sum loses its accuracy."""
    order = 40
    q = [Fraction(i, 64) for i in range(65)]
    rho = np.array([float(value) for value in q])
    theta = np.zeros_like(rho)

    result = _zernike_recursive_up_to_order(rho, theta, order, [0], [0])[0]

    for n in range(order + 1):
        for m in range(n % 2, n + 1, 2):
            s = (n - m) // 2
            exact = np.array([
                float(sum((-1)**k * math.comb(n - k, k) * math.comb(n - 2 * k, s - k) * value**(n - 2 * k) for k in range(s + 1)))
                for value in q
            ])
            j = (n * (n + 2) + m) // 2
            error = np.max(np.abs(result[j] - exact))
            assert error < 1e-12, f"Error {error:.2e} for R_{n}^{m} at order {order}."


def test_recursive_dimensions():
    """Test that the recursion returns results with the correct dimensions for N-D inputs."""
    Rho, Theta = _polar_grid(10, 12)
    order = 5
    n_polynomials = zernike_polynomials_count(order)

    result = _zernike_recursive_up_to_order(Rho, Theta, order, [0, 1, 0], [0, 0, 1])
    assert result.shape == (3, n_polynomials) + Rho.shape, f"Unexpected shape {result.shape}."
    assert not np.any(np.isnan(result)), "Results contain NaN values."

    # Consistency check with flattened inputs
    flat_result = _zernike_recursive_up_to_order(Rho.flatten(), Theta.flatten(), order, [0, 1, 0], [0, 0, 1])
    assert flat_result.shape == (3, n_polynomials, Rho.size), f"Unexpected flattened shape {flat_result.shape}."
    assert np.allclose(result.reshape(flat_result.shape), flat_result), "Mismatch between N-D and flattened results."

    # No derivative pair
    empty_result = _zernike_recursive_up_to_order(Rho, Theta, order, [], [])
    assert empty_result.shape == (0, n_polynomials) + Rho.shape, f"Unexpected shape {empty_result.shape} without derivative pair."


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_recursive_dtype(dtype):
    """Test that the recursion keeps the floating point dtype of the inputs and matches the explicit sum."""
    Rho, Theta = _polar_grid(dtype=dtype)
    order = 6

    recursive = _zernike_recursive_up_to_order(Rho, Theta, order, [0, 1], [0, 1])
    reference = _core_up_to_order(Rho, Theta, order, [0, 1], [0, 1])

    assert recursive.dtype == dtype, f"Expected dtype {dtype}, got {recursive.dtype}."
    rtol = 1e-4 if dtype == np.float32 else 1e-10
    _assert_close(recursive.astype(np.float64), reference.astype(np.float64), f"Mismatch for dtype {dtype}.", rtol=rtol)