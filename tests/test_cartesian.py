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

from pyzernike import cartesian_to_elliptic_annulus


ELLIPSE = dict(Rx=1.3, Ry=0.8, x0=0.1, y0=-0.2, alpha=0.7, h=0.25)


def _points():
    rng = numpy.random.default_rng(0)
    return rng.uniform(-1.0, 1.0, 50), rng.uniform(-1.0, 1.0, 50)


@pytest.mark.parametrize("x_derivative, y_derivative", [(1, 0), (0, 1), (2, 0), (1, 1), (0, 2)])
def test_derivatives_finite_differences(x_derivative, y_derivative):
    """The derivatives of order 1 and 2 match the centered finite differences of the lower order."""
    x, y = _points()
    eps = 1e-6
    rho, theta = cartesian_to_elliptic_annulus(x, y, x_derivative=x_derivative, y_derivative=y_derivative, **ELLIPSE)
    if x_derivative > 0:
        low = cartesian_to_elliptic_annulus(x - eps, y, x_derivative=x_derivative - 1, y_derivative=y_derivative, **ELLIPSE)
        high = cartesian_to_elliptic_annulus(x + eps, y, x_derivative=x_derivative - 1, y_derivative=y_derivative, **ELLIPSE)
    else:
        low = cartesian_to_elliptic_annulus(x, y - eps, x_derivative=x_derivative, y_derivative=y_derivative - 1, **ELLIPSE)
        high = cartesian_to_elliptic_annulus(x, y + eps, x_derivative=x_derivative, y_derivative=y_derivative - 1, **ELLIPSE)
    for value, l, u in zip((rho, theta), low, high):
        reference = (u - l) / (2 * eps)
        assert numpy.allclose(value, reference, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("x_derivative, y_derivative", [(3, 0), (0, 3), (2, 1), (1, 2), ([0, 1, 2], [0, 0, 1])])
def test_order_greater_than_two(x_derivative, y_derivative):
    """A total order of derivation greater than 2 raises a ValueError."""
    x, y = _points()
    with pytest.raises(ValueError):
        cartesian_to_elliptic_annulus(x, y, x_derivative=x_derivative, y_derivative=y_derivative, **ELLIPSE)


def test_order_two_sequence():
    """Several derivatives of total order lower than or equal to 2 are accepted together."""
    x, y = _points()
    rho, theta = cartesian_to_elliptic_annulus(x, y, x_derivative=[0, 1, 0, 2, 1, 0], y_derivative=[0, 0, 1, 0, 1, 2], **ELLIPSE)
    assert len(rho) == 6 and len(theta) == 6