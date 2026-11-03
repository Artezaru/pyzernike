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

from numbers import Integral
from typing import Sequence, Union

import numpy

from ..core.parsing import _as_osa_coefficients, _check_boolean, _check_resolution
from ..core.core_coefficients import _core_peak_to_valley


def zernike_peak_to_valley(
    coefficients: Union[Sequence[float], numpy.ndarray],
    *,
    normalize: bool = False,
    resolution: Integral = 256,
) -> float:
    r"""
    Compute the peak-to-valley (PV) of a sum of Zernike polynomials on the unit disk from its coefficients.

    .. math::

        PV = \max_{\rho \leq 1} f - \min_{\rho \leq 1} f
        \quad \text{with} \quad
        f = \sum_j c_j Z_j

    There is no closed-form expression: the function is evaluated with the recurrence relation on a polar grid of
    ``resolution`` radii in :math:`[0, 1]` and ``2 * resolution`` angles. The result is a lower bound of the exact
    value, which converges quickly with the resolution for smooth functions (the error is of the order of the
    variation of :math:`f` between two points of the grid).

    It is also the PV on an ellipse for the polynomials of
    :func:`pyzernike.xy_zernike_polynomial` with :math:`h = 0` (the image of the disk is the ellipse).

    .. seealso::

        - :func:`pyzernike.zernike_rms` for the root mean square.
        - :func:`pyzernike.zernike_evaluate` to evaluate the function on any grid.
        - The page :doc:`../../mathematical_description/coefficients` for the statistics of a sum of polynomials.

    Parameters
    ----------
    coefficients : Union[Sequence[float], numpy.ndarray]
        The coefficients :math:`c_j` in the OSA/ANSI ordering, as a 1D array of :math:`P = \frac{(N+1)(N+2)}{2}`
        finite real numbers.

    normalize : bool, optional
        If True, the coefficients are those of the polynomials normalized by the factors of Noll
        (see :func:`pyzernike.zernike_normalization`). Default is False.

    resolution : Integral, optional
        The number of radii of the polar grid (the number of angles is ``2 * resolution``). Must be an integer
        greater than or equal to 2. Default is 256.

    Returns
    -------
    float
        The peak-to-valley, in the unit of the coefficients.

    Raises
    ------
    TypeError
        If ``coefficients`` can not be converted to an array of real numbers, if ``normalize`` is not a boolean,
        or if ``resolution`` is not an integer.

    ValueError
        If ``coefficients`` is not a 1D array of finite values whose length is :math:`\frac{(N+1)(N+2)}{2}`, or if
        ``resolution`` is lower than 2.

    Examples
    --------
    The PV of the defocus :math:`Z_2^0 = 2\rho^2 - 1` is 2:

    .. code-block:: python

        from pyzernike import zernike_peak_to_valley

        zernike_peak_to_valley([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])  # 2.0

    """
    coefficients, order = _as_osa_coefficients(coefficients)
    normalize = _check_boolean(normalize, "normalize")
    resolution = _check_resolution(resolution)

    return _core_peak_to_valley(coefficients=coefficients, order=order, normalize=normalize, resolution=resolution)