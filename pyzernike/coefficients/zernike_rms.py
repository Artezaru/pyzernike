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

from typing import Sequence, Union

import numpy

from ..core.parsing import _as_osa_coefficients, _check_boolean
from ..core.core_coefficients import _core_rms


def zernike_rms(
    coefficients: Union[Sequence[float], numpy.ndarray],
    *,
    normalize: bool = False,
) -> float:
    r"""
    Compute the root mean square (RMS) of a sum of Zernike polynomials on the unit disk from its coefficients.

    The RMS is computed around the mean value of the function :math:`f = \sum_j c_j Z_j` (the piston is removed):

    .. math::

        \sigma^2 = \frac{1}{\pi} \int_{0}^{2\pi} \int_{0}^{1} \left( f - \bar{f} \right)^2 \rho \, d\rho \, d\theta

    The polynomials are orthogonal on the unit disk with :math:`\frac{1}{\pi} \int Z_j^2 = \frac{1}{N_j^2}`, where
    :math:`N_j` are the normalization factors of Noll, and only the piston has a non-zero mean, so:

    .. math::

        \sigma = \sqrt{\sum_{j > 0} \frac{c_j^2}{N_j^2}}
        \quad \text{or} \quad
        \sigma = \sqrt{\sum_{j > 0} c_j^2} \text{ for the normalized polynomials}

    The value is exact (no sampling). It is also the RMS on an ellipse for the polynomials of
    :func:`pyzernike.xy_zernike_polynomial` with :math:`h = 0` (the mapping onto the disk is affine).

    .. seealso::

        - :func:`pyzernike.zernike_peak_to_valley` for the peak-to-valley.
        - :func:`pyzernike.zernike_strehl_ratio` for the approximated Strehl ratio.
        - The page :doc:`../../mathematical_description/conventions` for the orthogonality of the Zernike polynomials.
        - The page :doc:`../../mathematical_description/coefficients` for the statistics of a sum of polynomials.

    Parameters
    ----------
    coefficients : Union[Sequence[float], numpy.ndarray]
        The coefficients :math:`c_j` in the OSA/ANSI ordering, as a 1D array of :math:`P = \frac{(N+1)(N+2)}{2}`
        finite real numbers.

    normalize : bool, optional
        If True, the coefficients are those of the polynomials normalized by the factors of Noll
        (see :func:`pyzernike.zernike_normalization`). Default is False.

    Returns
    -------
    float
        The RMS :math:`\sigma`, in the unit of the coefficients.

    Raises
    ------
    TypeError
        If ``coefficients`` can not be converted to an array of real numbers, or if ``normalize`` is not a boolean.

    ValueError
        If ``coefficients`` is not a 1D array of finite values whose length is :math:`\frac{(N+1)(N+2)}{2}`.

    Examples
    --------
    The RMS of the defocus :math:`Z_2^0 = 2\rho^2 - 1` is :math:`1 / \sqrt{3}`:

    .. code-block:: python

        import numpy
        from pyzernike import zernike_rms

        zernike_rms([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])  # 0.5773...
        zernike_rms([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], normalize=True)  # 1.0 (normalized defocus)

    """
    coefficients, order = _as_osa_coefficients(coefficients)
    normalize = _check_boolean(normalize, "normalize")

    return _core_rms(coefficients=coefficients, order=order, normalize=normalize)