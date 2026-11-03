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

import numpy
import pytest

import pyzernike.core.core_evaluate as core_evaluate
from pyzernike import (
    xy_zernike_evaluate,
    xy_zernike_polynomial_up_to_order,
    zernike_evaluate,
    zernike_polynomial_up_to_order,
    zernike_polynomials_count,
)


ORDER = 5
P = zernike_polynomials_count(ORDER)
ELLIPSE = dict(Rx=1.2, Ry=0.8, x0=0.1, y0=-0.05, alpha=0.4)


def _coefficients(seed=0):
    return numpy.random.default_rng(seed).normal(size=P)


def _polar_points():
    rho, theta = numpy.meshgrid(numpy.linspace(0, 1.1, 37), numpy.linspace(0, 2 * numpy.pi, 41))  # some rho > 1
    return rho, theta


def _xy_points():
    return numpy.meshgrid(numpy.linspace(-1.3, 1.4, 45), numpy.linspace(-1.1, 1.0, 39))


# ====================================================================== polar

@pytest.mark.parametrize("use_recursive", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_zernike_evaluate_matches_sum(use_recursive, normalize):
    """The evaluation is the sum of the polynomials weighted by the coefficients, for the values and derivatives."""
    rho, theta = _polar_points()
    c = _coefficients()
    rho_derivative, theta_derivative = [0, 1, 0, 2, 1], [0, 0, 1, 0, 2]
    Z = zernike_polynomial_up_to_order(
        rho, theta, order=ORDER, rho_derivative=rho_derivative, theta_derivative=theta_derivative,
        normalize=normalize, use_recursive=use_recursive, to_list=False,
    )
    expected = numpy.tensordot(c, Z, axes=(0, 1))
    result = zernike_evaluate(
        c, rho, theta, rho_derivative, theta_derivative, normalize=normalize, use_recursive=use_recursive, to_list=False,
    )
    assert result.shape == (5,) + rho.shape
    assert numpy.allclose(result, expected, atol=1e-11, equal_nan=True)
    assert numpy.all(numpy.isnan(result[:, rho > 1]))


def test_zernike_evaluate_output_formats():
    rho, theta = _polar_points()
    c = _coefficients()
    single = zernike_evaluate(c, rho, theta)
    assert isinstance(single, numpy.ndarray) and single.shape == rho.shape
    as_list = zernike_evaluate(c, rho, theta, rho_derivative=[0, 1])
    assert isinstance(as_list, list) and len(as_list) == 2
    assert numpy.allclose(as_list[0], single, atol=1e-13, equal_nan=True)
    assert zernike_evaluate(c, rho, theta, rho_derivative=[0, 1], to_list=False).shape == (2,) + rho.shape


def test_zernike_evaluate_default_and_dtype():
    rho, theta = _polar_points()
    c = _coefficients()
    result = zernike_evaluate(c, rho, theta, default=0.0)
    assert numpy.all(result[rho > 1] == 0.0)
    result32 = zernike_evaluate(c, rho.astype(numpy.float32), theta.astype(numpy.float32))
    assert result32.dtype == numpy.float32
    assert numpy.allclose(result32, zernike_evaluate(c, rho, theta), atol=1e-4, equal_nan=True)


@pytest.mark.parametrize("use_recursive", [False, True])
def test_chunks(monkeypatch, use_recursive):
    """The result does not depend on the size of the chunks of points."""
    rho, theta = _polar_points()
    x, y = _xy_points()
    c = _coefficients()
    polar = zernike_evaluate(c, rho, theta, [0, 1], [1, 0], use_recursive=use_recursive, to_list=False)
    xy = xy_zernike_evaluate(c, x, y, [0, 1], [1, 1], use_recursive=use_recursive, to_list=False, **ELLIPSE)
    monkeypatch.setattr(core_evaluate, "CHUNK_ELEMENTS", 7 * P)  # a few points per chunk
    assert numpy.allclose(zernike_evaluate(c, rho, theta, [0, 1], [1, 0], use_recursive=use_recursive, to_list=False), polar, equal_nan=True)
    assert numpy.allclose(xy_zernike_evaluate(c, x, y, [0, 1], [1, 1], use_recursive=use_recursive, to_list=False, **ELLIPSE), xy, equal_nan=True)


# ====================================================================== xy

@pytest.mark.parametrize("use_recursive", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("h", [0.0, 0.3])
def test_xy_zernike_evaluate_matches_sum(use_recursive, normalize, h):
    x, y = _xy_points()
    c = _coefficients()
    x_derivative, y_derivative = [0, 1, 0, 2, 1, 3], [0, 0, 1, 0, 1, 1]
    Z = xy_zernike_polynomial_up_to_order(
        x, y, ORDER, x_derivative, y_derivative, h=h, normalize=normalize, use_recursive=use_recursive, to_list=False, **ELLIPSE,
    )
    expected = numpy.tensordot(c, Z, axes=(0, 1))
    result = xy_zernike_evaluate(
        c, x, y, x_derivative, y_derivative, h=h, normalize=normalize, use_recursive=use_recursive, to_list=False, **ELLIPSE,
    )
    assert numpy.allclose(result, expected, atol=1e-9, equal_nan=True)
    assert numpy.isnan(result).any()  # points outside the annulus


def test_xy_zernike_evaluate_gradient_finite_differences():
    """The derivatives are consistent with the values (centered finite differences)."""
    x, y = numpy.meshgrid(numpy.linspace(-0.5, 0.5, 11), numpy.linspace(-0.4, 0.4, 9))
    c = _coefficients()
    eps = 1e-6
    dx, dy = xy_zernike_evaluate(c, x, y, [1, 0], [0, 1], **ELLIPSE)
    fd_x = (xy_zernike_evaluate(c, x + eps, y, **ELLIPSE) - xy_zernike_evaluate(c, x - eps, y, **ELLIPSE)) / (2 * eps)
    fd_y = (xy_zernike_evaluate(c, x, y + eps, **ELLIPSE) - xy_zernike_evaluate(c, x, y - eps, **ELLIPSE)) / (2 * eps)
    assert numpy.allclose(dx, fd_x, atol=1e-6)
    assert numpy.allclose(dy, fd_y, atol=1e-6)


# ====================================================================== errors

@pytest.mark.parametrize("evaluate", ["polar", "xy"])
def test_invalid_coefficients(evaluate):
    a, b = numpy.zeros(4), numpy.zeros(4)
    call = (lambda c: zernike_evaluate(c, a, b)) if evaluate == "polar" else (lambda c: xy_zernike_evaluate(c, a, b))
    for bad in (numpy.ones(7), numpy.ones(2), numpy.ones((2, 3)), [], [0.0, numpy.nan, 0.0]):
        with pytest.raises(ValueError):
            call(bad)
    with pytest.raises(TypeError):
        call(["a", "b", "c"])
    call(numpy.ones(1))  # order 0
    call(numpy.ones(15))  # order 4


def test_invalid_parameters():
    rho, theta = numpy.zeros(4), numpy.zeros(4)
    c = numpy.ones(6)
    with pytest.raises(ValueError):
        zernike_evaluate(c, rho, theta[:2])
    with pytest.raises(ValueError):
        zernike_evaluate(c, rho, theta, rho_derivative=-1)
    with pytest.raises(ValueError):
        zernike_evaluate(c, rho, theta, rho_derivative=[0, 1], theta_derivative=[0, 1, 2])
    with pytest.raises(TypeError):
        zernike_evaluate(c, rho, theta, default=True)
    for name in ("precompute", "use_recursive", "normalize", "to_list"):
        with pytest.raises(TypeError):
            zernike_evaluate(c, rho, theta, **{name: 1})
        with pytest.raises(TypeError):
            xy_zernike_evaluate(c, rho, theta, **{name: 1})
    with pytest.raises(ValueError):
        xy_zernike_evaluate(c, rho, theta, Rx=0.0)
    with pytest.raises(ValueError):
        xy_zernike_evaluate(c, rho, theta, h=1.0)