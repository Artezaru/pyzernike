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

import math

import numpy
import pytest

from pyzernike import (
    zernike_coefficients_convert,
    zernike_coefficients_rotate,
    zernike_coefficients_transform,
    zernike_evaluate,
    zernike_index_to_order,
    zernike_normalization,
    zernike_order_to_index,
    zernike_peak_to_valley,
    zernike_polynomial,
    zernike_polynomials_count,
    zernike_rms,
    zernike_strehl_ratio,
)


ORDER = 5
P = zernike_polynomials_count(ORDER)
N_OSA, M_OSA = zernike_index_to_order(list(range(P)))
FACTORS = numpy.array(zernike_normalization(N_OSA, M_OSA))


def _coefficients(seed=0):
    return numpy.random.default_rng(seed).normal(size=P)


def _disk_points(n_points=400, seed=1):
    rng = numpy.random.default_rng(seed)
    return numpy.sqrt(rng.uniform(0, 1, n_points)), rng.uniform(0, 2 * numpy.pi, n_points)


def _sum(coefficients, indexing, rho, theta, normalized=False):
    """Direct sum of the polynomials in any indexing (reference independent of the conversion)."""
    first = 0 if indexing == "OSA" else 1
    n, m = zernike_index_to_order(list(range(first, first + len(coefficients))), indexing=indexing)
    Z = numpy.array(zernike_polynomial(rho, theta, n=n, m=m, normalize=normalized))
    return numpy.tensordot(coefficients, Z, axes=1)


# ====================================================================== conversion

@pytest.mark.parametrize("input_indexing", ["OSA", "Noll", "Fringe"])
@pytest.mark.parametrize("output_indexing", ["OSA", "Noll", "Fringe"])
@pytest.mark.parametrize("length", [6, 11, 13])
def test_convert_indexing_keeps_the_function(input_indexing, output_indexing, length):
    rho, theta = _disk_points()
    c = numpy.random.default_rng(length).normal(size=length)
    converted = zernike_coefficients_convert(c, input_indexing=input_indexing, output_indexing=output_indexing)
    assert numpy.allclose(_sum(converted, output_indexing, rho, theta), _sum(c, input_indexing, rho, theta), atol=1e-12)
    if input_indexing == output_indexing:
        assert len(converted) == length


def test_convert_lengths():
    """OSA and Noll outputs are complete up to the maximum order, the Fringe output ends at the largest index."""
    noll = numpy.ones(11)  # Noll 1..11, up to Z_4^0
    assert len(zernike_coefficients_convert(noll, input_indexing="Noll", output_indexing="OSA")) == 15
    osa = numpy.ones(10)  # all the polynomials up to the order 3
    fringe = zernike_coefficients_convert(osa, output_indexing="Fringe")
    assert len(fringe) == 11  # Z_3^3 and Z_3^-3 have the Fringe indices 10 and 11
    assert fringe[9 - 1] == 0.0  # Fringe 9 is Z_4^0, not in the input
    assert numpy.count_nonzero(fringe) == 10


def test_convert_normalization():
    c = _coefficients()
    normalized = zernike_coefficients_convert(c, output_normalized=True)
    assert numpy.allclose(normalized, c / FACTORS)
    assert numpy.allclose(zernike_coefficients_convert(normalized, input_normalized=True), c)
    rho, theta = _disk_points()
    assert numpy.allclose(_sum(normalized, "OSA", rho, theta, normalized=True), _sum(c, "OSA", rho, theta), atol=1e-12)
    # Normalization and indexing together
    noll = zernike_coefficients_convert(c, output_indexing="Noll", output_normalized=True)
    assert numpy.allclose(zernike_coefficients_convert(noll, input_indexing="noll", input_normalized=True), c)


def test_convert_errors():
    with pytest.raises(ValueError):
        zernike_coefficients_convert(numpy.ones(3), input_indexing="Zemax")
    with pytest.raises(TypeError):
        zernike_coefficients_convert(numpy.ones(3), output_indexing=1)
    with pytest.raises(TypeError):
        zernike_coefficients_convert(numpy.ones(3), input_normalized=1)
    with pytest.raises(ValueError):
        zernike_coefficients_convert(numpy.ones((2, 2)))
    with pytest.raises(ValueError):
        zernike_coefficients_convert([numpy.inf])


# ====================================================================== rotation

@pytest.mark.parametrize("angle", [0.3, -1.2, numpy.pi / 2])
def test_rotate_function(angle):
    """f'(rho, theta) = f(rho, theta + angle)."""
    rho, theta = _disk_points()
    c = _coefficients()
    rotated = zernike_coefficients_rotate(c, angle)
    assert numpy.allclose(zernike_evaluate(rotated, rho, theta), zernike_evaluate(c, rho, theta + angle), atol=1e-12)


def test_rotate_properties():
    c = _coefficients()
    rotated = zernike_coefficients_rotate(c, 0.7)
    symmetric = numpy.array(M_OSA) == 0
    assert numpy.array_equal(rotated[symmetric], c[symmetric])
    assert numpy.allclose(zernike_coefficients_rotate(rotated, -0.7), c)
    assert numpy.allclose(zernike_coefficients_rotate(c, 2 * numpy.pi), c)
    assert numpy.isclose(zernike_rms(rotated), zernike_rms(c))
    # Same relation for the normalized coefficients
    assert numpy.allclose(zernike_coefficients_rotate(c / FACTORS, 0.7), rotated / FACTORS)


def test_rotate_errors():
    with pytest.raises(ValueError):
        zernike_coefficients_rotate(numpy.ones(5), 0.1)
    with pytest.raises(TypeError):
        zernike_coefficients_rotate(numpy.ones(6), "0.1")
    with pytest.raises(ValueError):
        zernike_coefficients_rotate(numpy.ones(6), numpy.nan)


# ====================================================================== transformation

@pytest.mark.parametrize("scale, x0, y0, angle", [(0.5, 0.0, 0.0, 0.0), (0.3, 0.5, -0.2, 0.0), (0.8, 0.1, 0.1, 0.6), (1.5, 0.2, 0.0, -0.3)])
def test_transform_function(scale, x0, y0, angle):
    """f'(p') = f(q) with q = (x0, y0) + scale R(angle) p'."""
    rho, theta = _disk_points()
    c = _coefficients()
    transformed = zernike_coefficients_transform(c, scale=scale, x0=x0, y0=y0, angle=angle)
    u, v = rho * numpy.cos(theta), rho * numpy.sin(theta)
    x = x0 + scale * (math.cos(angle) * u - math.sin(angle) * v)
    y = y0 + scale * (math.sin(angle) * u + math.cos(angle) * v)
    inside = numpy.hypot(x, y) <= 1  # the reference is only defined inside the initial pupil
    expected = zernike_evaluate(c, numpy.hypot(x, y), numpy.arctan2(y, x))
    assert numpy.allclose(zernike_evaluate(transformed, rho, theta)[inside], expected[inside], atol=1e-11)


def test_transform_particular_cases():
    c = _coefficients()
    assert numpy.allclose(zernike_coefficients_transform(c), c, atol=1e-13)
    assert numpy.allclose(zernike_coefficients_transform(c, angle=0.4), zernike_coefficients_rotate(c, 0.4), atol=1e-13)
    # Z_2^0(0.5 rho) = 0.25 Z_2^0(rho) - 0.75
    defocus = numpy.zeros(6)
    defocus[4] = 1.0
    assert numpy.allclose(zernike_coefficients_transform(defocus, scale=0.5), [-0.75, 0, 0, 0, 0.25, 0], atol=1e-14)
    # Composition: a scale of 0.5 then 0.4 is a scale of 0.2
    twice = zernike_coefficients_transform(zernike_coefficients_transform(c, scale=0.5), scale=0.4)
    assert numpy.allclose(twice, zernike_coefficients_transform(c, scale=0.2), atol=1e-12)


def test_transform_normalize():
    c = _coefficients()
    transformed = zernike_coefficients_transform(c, scale=0.6, x0=0.2, y0=0.1, angle=0.3)
    normalized = zernike_coefficients_transform(c / FACTORS, scale=0.6, x0=0.2, y0=0.1, angle=0.3, normalize=True)
    assert numpy.allclose(normalized * FACTORS, transformed, atol=1e-12)


def test_transform_errors():
    c = numpy.ones(6)
    with pytest.raises(ValueError):
        zernike_coefficients_transform(c, scale=0.0)
    with pytest.raises(ValueError):
        zernike_coefficients_transform(c, scale=-1.0)
    with pytest.raises(ValueError):
        zernike_coefficients_transform(c, x0=numpy.inf)
    with pytest.raises(TypeError):
        zernike_coefficients_transform(c, y0=True)
    with pytest.raises(TypeError):
        zernike_coefficients_transform(c, normalize=0)
    with pytest.raises(ValueError):
        zernike_coefficients_transform(numpy.ones(4))


# ====================================================================== statistics

def test_rms():
    c = _coefficients()
    # Reference: standard deviation on a fine grid uniform in area
    x, y = numpy.meshgrid(numpy.linspace(-1, 1, 1001), numpy.linspace(-1, 1, 1001))
    values = zernike_evaluate(c, numpy.hypot(x, y), numpy.arctan2(y, x))
    assert numpy.isclose(zernike_rms(c), numpy.nanstd(values), rtol=2e-3)
    assert numpy.isclose(zernike_rms(c / FACTORS, normalize=True), zernike_rms(c))
    assert numpy.isclose(zernike_rms([0, 0, 0, 0, 1.0, 0]), 1 / math.sqrt(3))
    piston = numpy.zeros(P)
    piston[0] = 5.0
    assert zernike_rms(piston) == 0.0


def test_peak_to_valley():
    assert numpy.isclose(zernike_peak_to_valley([0, 0, 0, 0, 1.0, 0]), 2.0)
    c = _coefficients()
    x, y = numpy.meshgrid(numpy.linspace(-1, 1, 801), numpy.linspace(-1, 1, 801))
    values = zernike_evaluate(c, numpy.hypot(x, y), numpy.arctan2(y, x))
    reference = numpy.nanmax(values) - numpy.nanmin(values)
    assert numpy.isclose(zernike_peak_to_valley(c), reference, rtol=1e-3)
    assert numpy.isclose(zernike_peak_to_valley(c / FACTORS, normalize=True), zernike_peak_to_valley(c))
    assert zernike_peak_to_valley(c, resolution=8) <= zernike_peak_to_valley(c, resolution=512) + 1e-12


def test_strehl_ratio():
    c = _coefficients() * 0.01
    sigma = zernike_rms(c)
    assert numpy.isclose(zernike_strehl_ratio(c, wavelength=0.5), math.exp(-(2 * math.pi * sigma / 0.5) ** 2))
    assert numpy.isclose(zernike_strehl_ratio(c / FACTORS, wavelength=0.5, normalize=True), zernike_strehl_ratio(c, wavelength=0.5))
    assert zernike_strehl_ratio(numpy.zeros(P), wavelength=1.0) == 1.0
    assert 0.8 < zernike_strehl_ratio([0, 0, 0, 0, 1 / 14, 0], wavelength=1.0, normalize=True) < 0.82


def test_statistics_errors():
    with pytest.raises(ValueError):
        zernike_rms(numpy.ones(4))
    with pytest.raises(TypeError):
        zernike_rms(numpy.ones(3), normalize="no")
    with pytest.raises(ValueError):
        zernike_peak_to_valley(numpy.ones(3), resolution=1)
    with pytest.raises(TypeError):
        zernike_peak_to_valley(numpy.ones(3), resolution=10.5)
    with pytest.raises(ValueError):
        zernike_strehl_ratio(numpy.ones(3), wavelength=0.0)
    with pytest.raises(ValueError):
        zernike_strehl_ratio(numpy.ones(3), wavelength=numpy.inf)
    with pytest.raises(TypeError):
        zernike_strehl_ratio(numpy.ones(3), wavelength="1")


def test_order_to_index_consistency():
    """The OSA index used by the rotation for a pair (m, -m) matches zernike_order_to_index."""
    c = numpy.zeros(P)
    c[zernike_order_to_index(n=3, m=1)] = 1.0
    rotated = zernike_coefficients_rotate(c, numpy.pi / 2)  # cos(theta + pi/2) = -sin(theta)
    assert numpy.isclose(rotated[zernike_order_to_index(n=3, m=-1)], -1.0)