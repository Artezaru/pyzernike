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
from ..core.core_coefficients import _core_transform_coefficients


def zernike_coefficients_transform(
    coefficients: Union[Sequence[float], numpy.ndarray],
    *,
    scale: Real = 1.0,
    x0: Real = 0.0,
    y0: Real = 0.0,
    angle: Real = 0.0,
    normalize: bool = False,
) -> numpy.ndarray:
    r"""
    Express a sum of Zernike polynomials in a new pupil, scaled, decentered and rotated with respect to the initial
    pupil.

    The new pupil is the disk of center :math:`(x_0, y_0)` and radius :math:`s`, in the normalized coordinates of the
    initial pupil (whose radius is 1), with its axes rotated by the angle :math:`\phi`. The point :math:`p'` of the
    new unit disk is the point :math:`q` of the initial pupil:

    .. math::

        q = \begin{pmatrix} x_0 \\ y_0 \end{pmatrix} + s \begin{pmatrix} \cos\phi & -\sin\phi \\ \sin\phi & \cos\phi \end{pmatrix} p'

    and the output coefficients :math:`c'` describe the same function in the new pupil:

    .. math::

        \sum_j c'_j Z_j(p') = \sum_j c_j Z_j(q)

    This is the classical problem of the comparison of measurements made on pupils of different sizes or positions
    (e.g. a smaller pupil, :math:`s < 1`, or a decentered sub-aperture). As a polynomial of degree :math:`N` stays a
    polynomial of degree :math:`N`, the new coefficients are exact and use the same polynomials (up to the order
    :math:`N`). They are computed with a matrix :math:`c' = T c`, obtained by projection on the polynomials with an
    exact quadrature, and cached for the last parameters.

    .. note::

        If the new pupil is not inside the initial pupil (:math:`\sqrt{x_0^2 + y_0^2} + s > 1`), the function is
        extrapolated outside the initial pupil with its polynomial expression, which is usually not meaningful for
        measured data.

    .. seealso::

        - :func:`pyzernike.zernike_coefficients_rotate` for a rotation only (closed-form expression).
        - The page :doc:`../../mathematical_description/coefficients` for the change of pupil and the exact quadrature.
        - The page :doc:`../../mathematical_description/fitting` for the consequence of this property on the estimation of the parameters of a
          pupil.

    Parameters
    ----------
    coefficients : Union[Sequence[float], numpy.ndarray]
        The coefficients :math:`c_j` in the OSA/ANSI ordering, as a 1D array of :math:`P = \frac{(N+1)(N+2)}{2}`
        finite real numbers.

    scale : Real, optional
        The radius :math:`s` of the new pupil, in units of the radius of the initial pupil. Must be finite and
        strictly positive. Default is 1.0.

    x0 : Real, optional
        The x coordinate :math:`x_0` of the center of the new pupil, in units of the radius of the initial pupil.
        Must be finite. Default is 0.0.

    y0 : Real, optional
        The y coordinate :math:`y_0` of the center of the new pupil, in units of the radius of the initial pupil.
        Must be finite. Default is 0.0.

    angle : Real, optional
        The rotation angle :math:`\phi` of the axes of the new pupil in radians (counterclockwise). Must be finite.
        Default is 0.0.

    normalize : bool, optional
        If True, the input and output coefficients are those of the polynomials normalized by the factors of Noll
        (see :func:`pyzernike.zernike_normalization`). Default is False.

    Returns
    -------
    numpy.ndarray
        The coefficients :math:`c'_j` in the new pupil, as a 1D array of :math:`P` ``numpy.float64``.

    Raises
    ------
    TypeError
        If ``coefficients`` can not be converted to an array of real numbers, if ``scale``, ``x0``, ``y0`` or
        ``angle`` is not a real number, or if ``normalize`` is not a boolean.

    ValueError
        If ``coefficients`` is not a 1D array of finite values whose length is :math:`\frac{(N+1)(N+2)}{2}`, if a
        parameter is not finite, or if ``scale`` is not strictly positive.

    Examples
    --------
    A defocus measured on a pupil of radius 1 is seen on the central part of radius 0.5 as a smaller defocus
    (and a piston): :math:`Z_2^0(0.5 \rho) = 2 (0.5 \rho)^2 - 1 = 0.25 Z_2^0(\rho) - 0.75`.

    .. code-block:: python

        import numpy
        from pyzernike import zernike_coefficients_transform

        coefficients = numpy.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])  # Z_2^0 (OSA index 4)
        transformed = zernike_coefficients_transform(coefficients, scale=0.5)
        # transformed ~ [-0.75, 0, 0, 0, 0.25, 0]

    A decentered sub-aperture of radius 0.3 centered at (0.5, 0): the defocus gives a tilt in the sub-aperture.

    .. code-block:: python

        transformed = zernike_coefficients_transform(coefficients, scale=0.3, x0=0.5)

    """
    coefficients, order = _as_osa_coefficients(coefficients)
    scale = _as_real(scale, "scale", minimum=0.0, strict=True)
    x0 = _as_real(x0, "x0")
    y0 = _as_real(y0, "y0")
    angle = _as_real(angle, "angle")
    normalize = _check_boolean(normalize, "normalize")

    return _core_transform_coefficients(
        coefficients=coefficients,
        order=order,
        scale=scale,
        x0=x0,
        y0=y0,
        angle=angle,
        normalize=normalize,
    )