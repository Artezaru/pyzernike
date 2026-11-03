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

from pyzernike import zernike_normalization, zernike_polynomial, zernike_polynomial_up_to_order, zernike_index_to_order


def _disk_quadrature(order):
    """Quadrature points and weights, exact for the products of two Zernike polynomials of order <= order,
    for the inner product (1/pi) int_0^2pi int_0^1 f g rho drho dtheta."""
    x, w = np.polynomial.legendre.leggauss(order + 2)  # exact for polynomials of degree 2 * order + 1 in rho
    rho_nodes, rho_weights = (x + 1) / 2, w / 2 * (x + 1) / 2  # includes the factor rho
    n_theta = 2 * order + 2  # exact for the trigonometric polynomials of frequency <= 2 * order
    theta_nodes = 2 * np.pi * np.arange(n_theta) / n_theta
    Rho, Theta = np.meshgrid(rho_nodes, theta_nodes, indexing='ij')
    weights = np.outer(rho_weights, np.full(n_theta, 2 * np.pi / n_theta)) / np.pi
    return Rho, Theta, weights


def test_normalization_values():
    """Test the first normalization factors of Noll."""
    expected = {(0, 0): 1.0, (1, 1): 2.0, (1, -1): 2.0, (2, 0): np.sqrt(3), (2, -2): np.sqrt(6), (2, 2): np.sqrt(6),
                (3, 1): np.sqrt(8), (3, -3): np.sqrt(8), (4, 0): np.sqrt(5), (4, 4): np.sqrt(10)}
    for (n, m), value in expected.items():
        assert np.isclose(zernike_normalization(n, m), value), f"Wrong normalization factor for n={n}, m={m}."

    n = [n for n, _ in expected]
    m = [m for _, m in expected]
    assert np.allclose(zernike_normalization(n, m), list(expected.values()))
    assert np.allclose(zernike_normalization(np.array(n), np.array(m)), list(expected.values()))


@pytest.mark.parametrize("use_recursive", [False, True])
def test_orthonormality(use_recursive):
    """Test that the normalized polynomials are orthonormal on the unit disk, and that the non normalized ones are only orthogonal."""
    order = 10
    Rho, Theta, weights = _disk_quadrature(order)

    for normalize in [True, False]:
        Z = zernike_polynomial_up_to_order(Rho, Theta, order=order, normalize=normalize, use_recursive=use_recursive, to_list=False)[0]
        Z = Z.reshape(Z.shape[0], -1)
        gram = (Z * weights.ravel()) @ Z.T

        if normalize:
            assert np.allclose(gram, np.eye(Z.shape[0]), atol=1e-12), "The normalized polynomials are not orthonormal."
        else:
            n, m = zernike_index_to_order(list(range(Z.shape[0])))
            expected = np.diag(1.0 / np.square(zernike_normalization(n, m)))
            assert np.allclose(gram, expected, atol=1e-12), "The non normalized polynomials have wrong norms."


def test_zernike_polynomial_normalize():
    """Test that normalize=True multiplies the polynomials and their derivatives by the normalization factors."""
    rho = np.linspace(0, 1.2, 100)  # out of domain to test the default value
    theta = np.linspace(0, 2 * np.pi, 100)

    for n in range(8):
        for m in range(-n, n + 1, 2):
            kwargs = dict(rho=rho, theta=theta, n=[n, n, n], m=[m, m, m], rho_derivative=[0, 1, 0], theta_derivative=[0, 0, 2], to_list=False)
            normalized = zernike_polynomial(normalize=True, **kwargs)
            raw = zernike_polynomial(normalize=False, **kwargs)

            assert np.allclose(normalized, zernike_normalization(n, m) * raw, equal_nan=True), f"Mismatch for n={n}, m={m}."
            assert np.all(np.isnan(normalized[:, rho > 1])), "The default value must not be normalized."


@pytest.mark.parametrize("use_recursive", [False, True])
def test_up_to_order_normalize(use_recursive):
    """Test that zernike_polynomial_up_to_order with normalize=True matches zernike_polynomial with normalize=True."""
    rho = np.linspace(0, 1, 50)
    theta = np.linspace(0, 2 * np.pi, 50)

    result = zernike_polynomial_up_to_order(rho, theta, order=8, rho_derivative=[0, 1], theta_derivative=[0, 1], normalize=True, use_recursive=use_recursive, to_list=False)
    n, m = zernike_index_to_order(list(range(result.shape[1])))

    for d, (a, b) in enumerate([(0, 0), (1, 1)]):
        expected = zernike_polynomial(rho, theta, n, m, rho_derivative=a, theta_derivative=b, normalize=True, to_list=False)
        assert np.allclose(result[d], expected), f"Mismatch for the derivative ({a}, {b}) with use_recursive={use_recursive}."


def test_invalid_inputs():
    """Test that invalid inputs raise the expected errors."""
    with pytest.raises(ValueError):
        zernike_normalization(2, 1)  # n - m odd
    with pytest.raises(ValueError):
        zernike_normalization(2, 4)  # |m| > n
    with pytest.raises(ValueError):
        zernike_normalization([2, 3], [0])  # different lengths
    with pytest.raises(TypeError):
        zernike_normalization(2, [0])  # integer mixed with a sequence
    with pytest.raises(TypeError):
        zernike_normalization(2.0, 0)  # not an integer
    with pytest.raises(TypeError):
        zernike_polynomial(np.linspace(0, 1, 5), np.zeros(5), 2, 0, normalize=1)  # not a boolean
    with pytest.raises(TypeError):
        zernike_polynomial_up_to_order(np.linspace(0, 1, 5), np.zeros(5), 2, normalize="yes")  # not a boolean