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

from ..core.parsing import _as_osa_coefficients, _as_real
from ..core.core_coefficients import _core_rotate_coefficients


def zernike_coefficients_rotate(
    coefficients: Union[Sequence[float], numpy.ndarray],
    angle: Real,
) -> numpy.ndarray:
    r"""
    Express a sum of Zernike polynomials in a pupil frame rotated by an angle :math:`\phi`.

    The output coefficients :math:`c'` describe the same function in the rotated frame:

    .. math::

        \sum_j c'_j Z_j(\rho, \theta) = \sum_j c_j Z_j(\rho, \theta + \phi)

    i.e. the point of polar angle :math:`\theta` in the new frame is the point of polar angle :math:`\theta + \phi`
    in the initial frame. For each pair of polynomials :math:`Z_n^{m}` and :math:`Z_n^{-m}` (:math:`m > 0`), with the
    coefficients :math:`a` of :math:`R_n^m(\rho) \cos(m\theta)` and :math:`b` of :math:`R_n^m(\rho) \sin(m\theta)`:

    .. math::

        a' = a \cos(m\phi) + b \sin(m\phi)
        \quad \text{and} \quad
        b' = b \cos(m\phi) - a \sin(m\phi)

    The rotationally symmetric polynomials (:math:`m = 0`) are unchanged. The relation is exact, and holds for the
    normalized and the not normalized polynomials (the factors of Noll are the same for :math:`m` and :math:`-m`).
    This is the particular case of :func:`pyzernike.zernike_coefficients_transform` with ``angle=phi``.

    .. seealso::

        - :func:`pyzernike.zernike_coefficients_transform` for a scaled, decentered and rotated pupil.
        - The page :doc:`../../mathematical_description/coefficients` for the derivation of the rotation of the coefficients.

    Parameters
    ----------
    coefficients : Union[Sequence[float], numpy.ndarray]
        The coefficients :math:`c_j` in the OSA/ANSI ordering, as a 1D array of :math:`P = \frac{(N+1)(N+2)}{2}`
        finite real numbers.

    angle : Real
        The rotation angle :math:`\phi` of the new frame in radians (counterclockwise). Must be finite.

    Returns
    -------
    numpy.ndarray
        The coefficients :math:`c'_j` in the rotated frame, as a 1D array of :math:`P` ``numpy.float64``.

    Raises
    ------
    TypeError
        If ``coefficients`` can not be converted to an array of real numbers, or if ``angle`` is not a real number.

    ValueError
        If ``coefficients`` is not a 1D array of finite values whose length is :math:`\frac{(N+1)(N+2)}{2}`, or if
        ``angle`` is not finite.

    Examples
    --------
    An astigmatism :math:`Z_2^2` rotated by 45 degrees becomes the astigmatism :math:`Z_2^{-2}`:

    .. code-block:: python

        import numpy
        from pyzernike import zernike_coefficients_rotate

        coefficients = numpy.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # Z_2^2 (OSA index 5)
        rotated = zernike_coefficients_rotate(coefficients, angle=numpy.pi / 4)
        # rotated ~ [0, 0, 0, -1, 0, 0]: cos(2 (theta + pi/4)) = -sin(2 theta)

    """
    coefficients, order = _as_osa_coefficients(coefficients)
    angle = _as_real(angle, "angle")

    return _core_rotate_coefficients(coefficients=coefficients, order=order, angle=angle)