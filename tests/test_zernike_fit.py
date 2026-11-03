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

import warnings

import numpy
import pytest

from pyzernike import (
    cartesian_to_elliptic_annulus,
    xy_zernike_fit,
    xy_zernike_polynomial_up_to_order,
    zernike_fit,
    zernike_normalization,
    zernike_index_to_order,
    zernike_polynomial_up_to_order,
    zernike_polynomials_count,
)


ORDER = 4
P = zernike_polynomials_count(ORDER)
ELLIPSE = dict(Rx=1.2, Ry=0.8, x0=0.1, y0=-0.05, alpha=0.4)


def _true_coefficients(seed=1):
    return numpy.random.default_rng(seed).normal(size=P)


def _polar_samples(n_points=3000, seed=0):
    """Points uniform in area on the unit disk."""
    rng = numpy.random.default_rng(seed)
    return numpy.sqrt(rng.uniform(0, 1, n_points)), rng.uniform(0, 2 * numpy.pi, n_points)


def _xy_samples(n_points=60):
    """A regular grid that covers the ellipse (the points outside give NaN polynomials)."""
    x, y = numpy.meshgrid(numpy.linspace(-1.2, 1.4, n_points), numpy.linspace(-1.1, 1.0, n_points))
    return x, y


def _polar_values(rho, theta, coefficients, normalize=False):
    Z = zernike_polynomial_up_to_order(rho, theta, order=ORDER, normalize=normalize, to_list=False)[0]
    return numpy.tensordot(coefficients, Z, axes=1)


def _xy_values(x, y, coefficients, h=0.0, normalize=False):
    Z = xy_zernike_polynomial_up_to_order(x, y, ORDER, h=h, normalize=normalize, to_list=False, **ELLIPSE)[0]
    return numpy.tensordot(coefficients, Z, axes=1)


# ====================================================================== exact recovery

@pytest.mark.parametrize("use_recursive", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_zernike_fit_exact(use_recursive, normalize):
    """Noise-free values are recovered exactly."""
    rho, theta = _polar_samples()
    c = _true_coefficients()
    values = _polar_values(rho, theta, c, normalize=normalize)
    coefficients, residuals = zernike_fit(
        rho, theta, values, order=ORDER, use_recursive=use_recursive, normalize=normalize, return_residuals=True,
    )
    assert coefficients.shape == (P,)
    assert numpy.allclose(coefficients, c, atol=1e-10)
    assert numpy.max(numpy.abs(residuals)) < 1e-10


@pytest.mark.parametrize("use_recursive", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("h", [0.0, 0.3])
def test_xy_zernike_fit_exact(use_recursive, normalize, h):
    """Noise-free values on the elliptic annulus are recovered exactly, and the points outside are ignored."""
    x, y = _xy_samples()
    c = _true_coefficients()
    values = _xy_values(x, y, c, h=h, normalize=normalize)  # NaN outside the annulus
    assert numpy.isnan(values).any()
    coefficients, residuals = xy_zernike_fit(
        x, y, values, order=ORDER, h=h, use_recursive=use_recursive, normalize=normalize, return_residuals=True, **ELLIPSE,
    )
    assert numpy.allclose(coefficients, c, atol=1e-9)
    inside = numpy.isfinite(values)
    assert numpy.all(numpy.isnan(residuals[~inside]))
    assert numpy.max(numpy.abs(residuals[inside])) < 1e-9


def test_xy_zernike_fit_matches_zernike_fit():
    """xy_zernike_fit is zernike_fit on the equivalent polar coordinates of the annulus."""
    x, y = _xy_samples()
    values = numpy.sin(2 * x) * numpy.cos(y) + x * y  # not a polynomial of the basis
    rho, theta = cartesian_to_elliptic_annulus(x, y, h=0.2, **ELLIPSE)
    rho = numpy.where(rho <= 1, rho, numpy.nan)
    expected = zernike_fit(rho, theta, values, order=ORDER)
    assert numpy.allclose(xy_zernike_fit(x, y, values, order=ORDER, h=0.2, **ELLIPSE), expected, atol=1e-12)


# ====================================================================== options

def test_explicit_and_recursive_agree():
    rho, theta = _polar_samples()
    values = numpy.exp(rho * numpy.cos(theta))
    explicit = zernike_fit(rho, theta, values, order=8)
    recursive = zernike_fit(rho, theta, values, order=8, use_recursive=True)
    assert numpy.allclose(explicit, recursive, atol=1e-10)


@pytest.mark.parametrize("fit", ["polar", "xy"])
def test_normalize_scales_the_coefficients(fit):
    """With the normalized basis, the coefficients are divided by the factors of Noll (same fitted function)."""
    n, m = zernike_index_to_order(list(range(P)))
    factors = numpy.array(zernike_normalization(n, m))
    if fit == "polar":
        rho, theta = _polar_samples()
        values = numpy.exp(rho * numpy.cos(theta))
        raw = zernike_fit(rho, theta, values, order=ORDER)
        normalized = zernike_fit(rho, theta, values, order=ORDER, normalize=True)
    else:
        x, y = _xy_samples()
        values = numpy.exp(x - y)
        raw = xy_zernike_fit(x, y, values, order=ORDER, **ELLIPSE)
        normalized = xy_zernike_fit(x, y, values, order=ORDER, normalize=True, **ELLIPSE)
    assert numpy.allclose(normalized * factors, raw, atol=1e-10)


def test_variance_from_normalized_coefficients():
    """On a full ellipse sampled uniformly in area, the variance is the quadratic sum of the normalized coefficients."""
    x, y = numpy.meshgrid(numpy.linspace(-1.5, 1.5, 801), numpy.linspace(-1.5, 1.5, 801))
    c = _true_coefficients()
    values = _xy_values(x, y, c, normalize=True)
    coefficients = xy_zernike_fit(x, y, values, order=ORDER, normalize=True, **ELLIPSE)
    assert numpy.isclose(numpy.sqrt(numpy.sum(coefficients[1:] ** 2)), numpy.nanstd(values), rtol=1e-2)


def test_weights():
    """A zero weight removes a sample, and a low weight reduces the influence of an outlier."""
    rho, theta = _polar_samples()
    c = _true_coefficients()
    values = _polar_values(rho, theta, c)
    values[0] += 100.0  # outlier

    weights = numpy.ones_like(values)
    weights[0] = 0.0
    assert numpy.allclose(zernike_fit(rho, theta, values, order=ORDER, weights=weights), c, atol=1e-10)

    weights[0] = 1e-6
    error_weighted = numpy.abs(zernike_fit(rho, theta, values, order=ORDER, weights=weights) - c).max()
    error_unweighted = numpy.abs(zernike_fit(rho, theta, values, order=ORDER) - c).max()
    assert error_weighted < 1e-3 * error_unweighted


def test_invalid_samples_are_ignored():
    """The samples with rho outside [0, 1] or non-finite values are ignored, and their residuals are NaN."""
    rho, theta = _polar_samples()
    c = _true_coefficients()
    values = _polar_values(rho, theta, c)
    rho = rho.copy()
    rho[:10] = 1.5
    values[10:20] = numpy.nan
    theta = theta.copy()
    theta[20:30] = numpy.inf
    coefficients, residuals = zernike_fit(rho, theta, values, order=ORDER, return_residuals=True)
    assert numpy.allclose(coefficients, c, atol=1e-10)
    assert numpy.all(numpy.isnan(residuals[:30]))
    assert numpy.all(numpy.isfinite(residuals[30:]))


def test_noise():
    """With noise, the coefficients are close to the true ones and the residuals have the level of the noise."""
    rng = numpy.random.default_rng(3)
    rho, theta = _polar_samples(n_points=20000)
    c = _true_coefficients()
    values = _polar_values(rho, theta, c, normalize=True) + rng.normal(scale=0.01, size=rho.shape)
    coefficients, residuals = zernike_fit(rho, theta, values, order=ORDER, normalize=True, return_residuals=True)
    assert numpy.abs(coefficients - c).max() < 2e-3
    assert numpy.isclose(numpy.std(residuals), 0.01, rtol=0.05)


def test_multidimensional_inputs():
    """The samples can have any shape, and the residuals keep it."""
    rho, theta = _polar_samples(n_points=3000)
    values = numpy.cos(3 * rho) * numpy.sin(theta)
    flat = zernike_fit(rho, theta, values, order=ORDER)
    coefficients, residuals = zernike_fit(
        rho.reshape(30, 100), theta.reshape(30, 100), values.reshape(30, 100), order=ORDER, return_residuals=True,
    )
    assert numpy.allclose(coefficients, flat, atol=1e-12)
    assert residuals.shape == (30, 100)


# ====================================================================== warnings and errors

@pytest.mark.parametrize("fit", ["polar", "xy"])
def test_rank_deficient_warning(fit):
    """Samples on a single circle do not determine all the coefficients."""
    theta = numpy.linspace(0, 2 * numpy.pi, 200, endpoint=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if fit == "polar":
            zernike_fit(numpy.full(200, 0.5), theta, numpy.cos(theta), order=ORDER)
        else:
            xy_zernike_fit(0.5 * numpy.cos(theta), 0.5 * numpy.sin(theta), numpy.cos(theta), order=ORDER)
    assert any(issubclass(w.category, RuntimeWarning) and "rank deficient" in str(w.message) for w in caught)


def test_zernike_fit_invalid_inputs():
    rho, theta = _polar_samples(n_points=100)
    values = rho.copy()
    with pytest.raises(ValueError):
        zernike_fit(rho, theta, values[:50], order=ORDER)  # shape
    with pytest.raises(ValueError):
        zernike_fit(rho, theta, values, order=-1)
    with pytest.raises(TypeError):
        zernike_fit(rho, theta, values, order=2.5)
    with pytest.raises(ValueError):
        zernike_fit(rho, theta, values, order=ORDER, weights=-numpy.ones(100))
    with pytest.raises(ValueError):
        zernike_fit(rho, theta, values, order=ORDER, weights=numpy.ones(50))
    with pytest.raises(ValueError):
        zernike_fit(rho[:10], theta[:10], values[:10], order=ORDER)  # 10 samples < 15 polynomials
    for name in ("use_recursive", "normalize", "return_residuals"):
        with pytest.raises(TypeError):
            zernike_fit(rho, theta, values, order=ORDER, **{name: 1})


def test_xy_zernike_fit_invalid_inputs():
    x, y = _xy_samples(n_points=20)
    values = x + y
    with pytest.raises(ValueError):
        xy_zernike_fit(x, y, values[:10], order=ORDER)
    with pytest.raises(ValueError):
        xy_zernike_fit(x, y, values, order=ORDER, Rx=-1.0)
    with pytest.raises(ValueError):
        xy_zernike_fit(x, y, values, order=ORDER, h=1.0)
    with pytest.raises(ValueError):
        xy_zernike_fit(x, y, values, order=ORDER, x0=numpy.inf)
    with pytest.raises(TypeError):
        xy_zernike_fit(x, y, values, order=ORDER, alpha="0")
    with pytest.raises(ValueError):
        xy_zernike_fit(x, y, values, order=ORDER, weights=-numpy.ones_like(x))
    with pytest.raises(ValueError):
        xy_zernike_fit(x, y, values, order=ORDER, Rx=0.01, Ry=0.01)  # almost no point inside the ellipse
    with pytest.raises(TypeError):
        xy_zernike_fit(x, y, values, order=ORDER, normalize="yes")