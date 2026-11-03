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

from numbers import Integral, Real
from typing import Optional, Sequence, Union

import numpy

from ..core.parsing import _as_ellipse_parameters, _broadcast_indices, _check_boolean, _check_resolution
from .core_xy_display import _core_xy_display


def xy_zernike_display(
    n: Union[Integral, Sequence[Integral], numpy.ndarray],
    m: Union[Integral, Sequence[Integral], numpy.ndarray],
    x_derivative: Optional[Union[Integral, Sequence[Integral], numpy.ndarray]] = 0,
    y_derivative: Optional[Union[Integral, Sequence[Integral], numpy.ndarray]] = 0,
    *,
    Rx: Optional[Real] = 1.0,
    Ry: Optional[Real] = 1.0,
    x0: Optional[Real] = 0.0,
    y0: Optional[Real] = 0.0,
    alpha: Optional[Real] = 0.0,
    h: Optional[Real] = 0.0,
    precompute: bool = True,
    resolution: int = 200,
) -> None:
    r"""
    Display the Zernike polynomials on an elliptic annulus [1]_ and their derivatives with respect to :math:`x` and
    :math:`y` in a ``matplotlib`` figure.

    .. figure:: ../../../pyzernike/resources/elliptic_annulus_domain.png
        :width: 400px
        :align: center

        The parameters of the elliptic annulus domain.

    Each polynomial is drawn as a colormap on a Cartesian grid covering the elliptic annulus, transparent
    outside the annulus, with the boundaries of the annulus. When several polynomials are given, the
    "Previous" and "Next" buttons of the figure navigate through them. The color scale spans
    :math:`[\min(-1, Z), \max(1, Z)]`.

    The polynomials and their derivatives are computed as in :func:`pyzernike.xy_zernike_polynomial` (any
    derivative order, exact at the center of the ellipse). If :math:`|m| > n` or :math:`n - m` is odd, the
    displayed polynomial is zero.

    **Shape of the inputs.** ``n``, ``m``, ``x_derivative`` and ``y_derivative`` can each be a single integer
    or a sequence of integers (list, tuple or 1D numpy array). The integers are repeated to the length of the
    sequences, which must all have the same length.

    .. seealso::

        - :func:`pyzernike.xy_zernike_polynomial` for the numerical evaluation of the polynomials on the elliptic annulus.
        - :func:`pyzernike.xy_display_interactive` to explore the polynomials and the parameters of the annulus.
        - :func:`pyzernike.zernike_display` to display the Zernike polynomials on the unit disk.
        - The page :doc:`../../mathematical_description/elliptic_annulus` for the extension of the Zernike polynomials to the elliptic annulus.

    Parameters
    ----------
    n : Union[Integral, Sequence[Integral], numpy.ndarray]
        The radial order(s) :math:`n`.

    m : Union[Integral, Sequence[Integral], numpy.ndarray]
        The azimuthal frequency(ies) :math:`m`.

    x_derivative : Optional[Union[Integral, Sequence[Integral], numpy.ndarray]], optional
        The non-negative order(s) of the derivative with respect to :math:`x`. Default is 0 (no derivative).

    y_derivative : Optional[Union[Integral, Sequence[Integral], numpy.ndarray]], optional
        The non-negative order(s) of the derivative with respect to :math:`y`. Default is 0 (no derivative).

    Rx : Optional[Real], optional
        The semi-axis :math:`R_x` of the ellipse along its x axis. Must be finite and strictly positive. Default is 1.0.

    Ry : Optional[Real], optional
        The semi-axis :math:`R_y` of the ellipse along its y axis. Must be finite and strictly positive. Default is 1.0.

    x0 : Optional[Real], optional
        The x coordinate :math:`x_0` of the center of the ellipse. Must be finite. Default is 0.0.

    y0 : Optional[Real], optional
        The y coordinate :math:`y_0` of the center of the ellipse. Must be finite. Default is 0.0.

    alpha : Optional[Real], optional
        The rotation angle :math:`\alpha` of the ellipse in radians. Must be finite. Default is 0.0.

    h : Optional[Real], optional
        The ratio :math:`h` of the inner to the outer semi-axes, in :math:`[0, 1)`. Default is 0.0 (full ellipse).

    precompute : bool, optional
        If True, the powers of :math:`\rho_{eq}` and :math:`1/r` and the cosine and sine terms are cached and
        shared between the polynomials. Default is True.

    resolution : int, optional
        The number of points along each axis of the display grid. Default is 200.

    None is accepted for all the parameters of the ellipse and for the derivative orders, and means the default value.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If ``n``, ``m``, ``x_derivative`` or ``y_derivative`` is not an integer or a sequence of integers, if a
        parameter of the ellipse is not a real number, if ``precompute`` is not a boolean, or if ``resolution``
        is not an integer.

    ValueError
        If the sequences do not have the same length or are empty, if a derivative order is negative, if a
        parameter of the ellipse is not finite, if ``Rx`` or ``Ry`` is not strictly positive, if ``h`` is not
        in :math:`[0, 1)`, or if ``resolution`` is lower than 2.

    References
    ----------
    .. [1] R. Navarro, J. L. López, J. A. Díaz and E. Pérez Sinusía, "Generalization of Zernike polynomials for
       regular portions of circles and ellipses," Optics Express 22(18), 21263-21279 (2014).
       https://doi.org/10.1364/OE.22.021263

    Examples
    --------
    Display the defocus :math:`Z_{2}^{0}` on a rotated ellipse:

    .. code-block:: python

        from pyzernike import xy_zernike_display

        xy_zernike_display(n=2, m=0, Rx=2.0, Ry=1.0, alpha=0.5)

    Display the gradient of the astigmatism :math:`Z_{2}^{2}` on an elliptic annulus:

    .. code-block:: python

        xy_zernike_display(n=2, m=2, x_derivative=[1, 0], y_derivative=[0, 1], Rx=2.0, Ry=1.0, h=0.3)

    """
    Rx, Ry, x0, y0, alpha, h = _as_ellipse_parameters(Rx, Ry, x0, y0, alpha, h)
    precompute = _check_boolean(precompute, "precompute")
    resolution = _check_resolution(resolution)
    indices, _ = _broadcast_indices(
        integers={"n": n, "m": m},
        orders={"x_derivative": x_derivative, "y_derivative": y_derivative},
    )
    if not indices["n"]:
        raise ValueError("At least one polynomial must be given.")

    _core_xy_display(
        n=indices["n"],
        m=indices["m"],
        x_derivative=indices["x_derivative"],
        y_derivative=indices["y_derivative"],
        Rx=Rx,
        Ry=Ry,
        x0=x0,
        y0=y0,
        alpha=alpha,
        h=h,
        precompute=precompute,
        resolution=resolution,
    )