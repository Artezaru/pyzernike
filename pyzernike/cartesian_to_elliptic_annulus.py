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
from typing import List, Optional, Sequence, Tuple, Union

import numpy

from .core.parsing import _as_common_floating_arrays, _as_ellipse_parameters, _broadcast_integers
from .core.cartesian_to_elliptic_annulus import _cartesian_to_elliptic_annulus


def cartesian_to_elliptic_annulus(
    x: numpy.ndarray,
    y: numpy.ndarray,
    Rx: Optional[Real] = 1.0,
    Ry: Optional[Real] = 1.0,
    x0: Optional[Real] = 0.0,
    y0: Optional[Real] = 0.0,
    alpha: Optional[Real] = 0.0,
    h: Optional[Real] = 0.0,
    x_derivative: Optional[Union[Integral, Sequence[Integral], numpy.ndarray]] = 0,
    y_derivative: Optional[Union[Integral, Sequence[Integral], numpy.ndarray]] = 0,
    *,
    to_list: bool = True,
) -> Union[Tuple[numpy.ndarray, numpy.ndarray], Tuple[List[numpy.ndarray], List[numpy.ndarray]]]:
    r"""
    Transform Cartesian coordinates :math:`(x, y)` into the equivalent polar coordinates
    :math:`(\rho_{eq}, \theta_{eq})` of an elliptic annulus, and compute their derivatives with
    respect to :math:`x` and :math:`y`.

    .. figure:: ../../../pyzernike/resources/elliptic_annulus_domain.png
        :width: 400px
        :align: center

        The parameters of the elliptic annulus domain.

    The elliptic annulus is defined by its semi-axes :math:`R_x` and :math:`R_y`, its center
    :math:`(x_0, y_0)`, its rotation angle :math:`\alpha` and the ratio :math:`h = \frac{a}{R_x} = \frac{b}{R_y}`
    of the inner to the outer semi-axes. The coordinates are first expressed in the frame of the ellipse:

    .. math::

        X = \cos(\alpha) (x - x_0) + \sin(\alpha) (y - y_0)
        \quad \text{and} \quad
        Y = -\sin(\alpha) (x - x_0) + \cos(\alpha) (y - y_0)

    then mapped onto the unit annulus [1]_:

    .. math::

        r = \sqrt{\left(\frac{X}{R_x}\right)^2 + \left(\frac{Y}{R_y}\right)^2},
        \quad
        \rho_{eq} = \frac{r - h}{1 - h}
        \quad \text{and} \quad
        \theta_{eq} = \operatorname{atan2}\left(\frac{Y}{R_y}, \frac{X}{R_x}\right) \in [-\pi, \pi]

    The points of the elliptic annulus are those with :math:`0 \leq \rho_{eq} \leq 1`. The coordinates
    :math:`(\rho_{eq}, \theta_{eq})` can be passed directly to :func:`pyzernike.zernike_polynomial`.

    The derivatives of total order 1 and 2 are computed with the analytical expressions obtained by the
    chain rule (see the page :doc:`../../mathematical_description`). The derivatives of total order 3 and
    more are computed by symbolic differentiation with ``sympy``, which has a high computational cost.

    **Shape of the inputs and of the output.** ``x_derivative`` and ``y_derivative`` can each be a single
    integer or a sequence of integers (list, tuple or 1D numpy array):

    - If both are integers, the output is a tuple ``(rho_eq, theta_eq)`` of two arrays with the shape of ``x``.
    - If at least one of them is a sequence, the integer is repeated to the length :math:`N` of the
      sequence (the sequences must have the same length). The output is a tuple of two lists of
      :math:`N` arrays (``to_list=True``), or of two arrays of shape :math:`(N, \dots)` (``to_list=False``).

    .. seealso::

        - :func:`pyzernike.xy_zernike_polynomial` to compute the Zernike polynomials on the elliptic annulus directly.
        - :func:`pyzernike.zernike_polynomial` to compute the Zernike polynomials in polar coordinates.
        - The page :doc:`../../mathematical_description` for the extension of the Zernike polynomials to the elliptic annulus.

    Parameters
    ----------
    x : numpy.ndarray
        The x Cartesian coordinates with any shape ``(...)``.

    y : numpy.ndarray
        The y Cartesian coordinates with the same shape as ``x``.
        If ``x`` and ``y`` are floating point arrays of the same dtype (e.g. ``numpy.float32``), this
        dtype is kept. Otherwise, both are converted to ``numpy.float64``.

    Rx : Optional[Real], optional
        The semi-axis :math:`R_x` of the ellipse along its x axis. Must be strictly positive. Default is 1.0.

    Ry : Optional[Real], optional
        The semi-axis :math:`R_y` of the ellipse along its y axis. Must be strictly positive. Default is 1.0.

    x0 : Optional[Real], optional
        The x coordinate :math:`x_0` of the center of the ellipse. Default is 0.0.

    y0 : Optional[Real], optional
        The y coordinate :math:`y_0` of the center of the ellipse. Default is 0.0.

    alpha : Optional[Real], optional
        The rotation angle :math:`\alpha` of the ellipse in radians. Default is 0.0.

    h : Optional[Real], optional
        The ratio :math:`h` of the inner to the outer semi-axes, in :math:`[0, 1)`. Default is 0.0 (full ellipse).

    x_derivative : Optional[Union[Integral, Sequence[Integral], numpy.ndarray]], optional
        The order(s) of the derivative with respect to :math:`x`. Default is 0 (no derivative).

    y_derivative : Optional[Union[Integral, Sequence[Integral], numpy.ndarray]], optional
        The order(s) of the derivative with respect to :math:`y`. Default is 0 (no derivative).

    to_list : bool, optional
        Only used when several derivatives are computed. If True, the output is a tuple of two lists of
        arrays. If False, it is a tuple of two arrays of shape :math:`(N, \dots)`. Default is True.

    None is accepted for all the parameters with a default value and means this default value.

    Returns
    -------
    Union[Tuple[numpy.ndarray, numpy.ndarray], Tuple[List[numpy.ndarray], List[numpy.ndarray]]]
        The tuple ``(rho_eq, theta_eq)``:

        - If ``x_derivative`` and ``y_derivative`` are integers: two arrays with the shape of ``x``.
        - Otherwise, if ``to_list`` is True: two lists of :math:`N` arrays with the shape of ``x``.
        - Otherwise: two arrays of shape :math:`(N, \dots)`.

        ``rho_eq[i]`` (resp. ``theta_eq[i]``) is the derivative of :math:`\rho_{eq}` (resp.
        :math:`\theta_{eq}`) of order ``x_derivative[i]`` with respect to :math:`x` and
        ``y_derivative[i]`` with respect to :math:`y`.

    Raises
    ------
    TypeError
        If ``x`` or ``y`` can not be converted to a floating point array, if a parameter of the ellipse
        is not a real number, if ``x_derivative`` or ``y_derivative`` is not an integer or a sequence of
        integers, or if ``to_list`` is not a boolean.

    ValueError
        If ``x`` and ``y`` do not have the same shape, if ``Rx`` or ``Ry`` is not strictly positive, if
        ``h`` is not in :math:`[0, 1)`, if the sequences do not have the same length, or if a derivative
        order is negative.

    References
    ----------
    .. [1] R. Navarro et al., "Generalization of Zernike polynomials for regular portions of circles
       and ellipses," Optics Express 22(18), 21263-21279 (2014). https://doi.org/10.1364/OE.22.021263

    Examples
    --------
    Compute the defocus :math:`Z_{2}^{0}` on a circle of radius 2 centered at :math:`(1, 1)`:

    .. code-block:: python

        import numpy
        from pyzernike import cartesian_to_elliptic_annulus, zernike_polynomial

        X, Y = numpy.meshgrid(numpy.linspace(-1, 3, 100), numpy.linspace(-1, 3, 100))
        rho_eq, theta_eq = cartesian_to_elliptic_annulus(X, Y, Rx=2.0, Ry=2.0, x0=1.0, y0=1.0)
        Z20 = zernike_polynomial(rho_eq, theta_eq, n=2, m=0)

    The complete process can be done in one step with :func:`pyzernike.xy_zernike_polynomial`.

    Compute the coordinates and their first derivatives with respect to :math:`x` and :math:`y`:

    .. code-block:: python

        rho_eq, theta_eq = cartesian_to_elliptic_annulus(X, Y, Rx=2.0, Ry=2.0, x0=1.0, y0=1.0, x_derivative=[0, 1, 0], y_derivative=[0, 0, 1])
        rho, drho_dx, drho_dy = rho_eq
        theta, dtheta_dx, dtheta_dy = theta_eq

    """
    x, y = _as_common_floating_arrays(x=x, y=y)
    Rx, Ry, x0, y0, alpha, h = _as_ellipse_parameters(Rx, Ry, x0, y0, alpha, h)
    if not isinstance(to_list, bool):
        raise TypeError("to_list must be a boolean.")

    if x_derivative is None:
        x_derivative = 0
    if y_derivative is None:
        y_derivative = 0

    derivatives, single = _broadcast_integers(x_derivative=x_derivative, y_derivative=y_derivative)
    if any(d < 0 for d in derivatives["x_derivative"] + derivatives["y_derivative"]):
        raise ValueError("x_derivative and y_derivative must contain non-negative integers.")

    float_type = x.dtype.type
    rho_eq, theta_eq = _cartesian_to_elliptic_annulus(
        x, y,
        float_type(Rx), float_type(Ry), float_type(x0), float_type(y0), float_type(alpha), float_type(h),
        derivatives["x_derivative"],
        derivatives["y_derivative"],
    )

    if single:
        return rho_eq[0], theta_eq[0]
    if to_list:
        return rho_eq, theta_eq
    return (
        numpy.stack([numpy.broadcast_to(value, x.shape) for value in rho_eq]) if rho_eq else numpy.empty((0,) + x.shape, dtype=x.dtype),
        numpy.stack([numpy.broadcast_to(value, x.shape) for value in theta_eq]) if theta_eq else numpy.empty((0,) + x.shape, dtype=x.dtype),
    )