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

from numbers import Real
from typing import Sequence, Union

import numpy

from ..core.parsing import _as_osa_coefficients, _as_real, _check_boolean
from ..core.core_coefficients import _core_rms, _core_strehl_ratio


def zernike_strehl_ratio(
    coefficients: Union[Sequence[float], numpy.ndarray],
    wavelength: Real,
    *,
    normalize: bool = False,
) -> float:
    r"""
    Approximate the Strehl ratio of a wavefront described by the coefficients of Zernike polynomials.

    The Strehl ratio :math:`S` is the ratio of the peak intensity of the point spread function to the peak intensity
    of the aberration-free system. For small aberrations, it depends only on the RMS :math:`\sigma` of the wavefront
    (see :func:`pyzernike.zernike_rms`), with the extended Maréchal approximation [1]_:

    .. math::

        S \approx \exp\left( -\left( \frac{2 \pi \sigma}{\lambda} \right)^2 \right)

    The approximation is accurate for :math:`S \gtrsim 0.3` (:math:`\sigma \lesssim \lambda / 6`). The Maréchal
    criterion of a diffraction-limited system, :math:`\sigma \leq \lambda / 14`, corresponds to :math:`S \geq 0.8`.

    .. seealso::

        - :func:`pyzernike.zernike_rms` for the RMS of the wavefront.
        - The page :doc:`../../mathematical_description/coefficients` for the statistics of a sum of polynomials.

    Parameters
    ----------
    coefficients : Union[Sequence[float], numpy.ndarray]
        The coefficients :math:`c_j` of the wavefront in the OSA/ANSI ordering, as a 1D array of
        :math:`P = \frac{(N+1)(N+2)}{2}` finite real numbers.

    wavelength : Real
        The wavelength :math:`\lambda`, in the unit of the coefficients (e.g. both in micrometers, or the
        coefficients in waves and ``wavelength=1``). Must be finite and strictly positive.

    normalize : bool, optional
        If True, the coefficients are those of the polynomials normalized by the factors of Noll
        (see :func:`pyzernike.zernike_normalization`). Default is False.

    Returns
    -------
    float
        The approximated Strehl ratio :math:`S \in (0, 1]`.

    Raises
    ------
    TypeError
        If ``coefficients`` can not be converted to an array of real numbers, if ``wavelength`` is not a real number,
        or if ``normalize`` is not a boolean.

    ValueError
        If ``coefficients`` is not a 1D array of finite values whose length is :math:`\frac{(N+1)(N+2)}{2}`, or if
        ``wavelength`` is not finite or not strictly positive.

    References
    ----------
    .. [1] V. N. Mahajan, "Strehl ratio for primary aberrations in terms of their aberration variance," Journal of
       the Optical Society of America 73(6), 860-861 (1983). https://doi.org/10.1364/JOSA.73.000860

    Examples
    --------
    A normalized defocus of :math:`\lambda / 14` RMS gives a Strehl ratio of about 0.82:

    .. code-block:: python

        from pyzernike import zernike_strehl_ratio

        zernike_strehl_ratio([0.0, 0.0, 0.0, 0.0, 1 / 14, 0.0], wavelength=1.0, normalize=True)  # 0.8176...

    """
    coefficients, order = _as_osa_coefficients(coefficients)
    wavelength = _as_real(wavelength, "wavelength", minimum=0.0, strict=True)
    normalize = _check_boolean(normalize, "normalize")

    rms = _core_rms(coefficients=coefficients, order=order, normalize=normalize)
    return _core_strehl_ratio(rms=rms, wavelength=wavelength)