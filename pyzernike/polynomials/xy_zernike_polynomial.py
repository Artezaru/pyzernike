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
from typing import List, Optional, Sequence, Union

import numpy

from ..core.parsing import (
    _as_common_floating_arrays,
    _as_default_value,
    _as_ellipse_parameters,
    _broadcast_indices,
    _check_boolean,
    _format_output,
)
from ..core.core_cartesian_to_elliptic_annulus import _core_cartesian_to_elliptic_annulus
from ..core.core_xy_polynomial import _core_xy_polynomials


def xy_zernike_polynomial(
    x: numpy.ndarray,
    y: numpy.ndarray,
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
    default: Union[int, float] = numpy.nan,
    precompute: bool = True,
    normalize: bool = False,
    to_list: bool = True,
) -> Union[numpy.ndarray, List[numpy.ndarray]]:
    r"""
    Compute the Zernike polynomials on an elliptic annulus for Cartesian coordinates :math:`(x, y)`, and
    their derivatives of any order with respect to :math:`x` and :math:`y`.

    .. figure:: ../../../pyzernike/resources/elliptic_annulus_domain.png
        :width: 400px
        :align: center

        The parameters of the elliptic annulus domain.

    The Zernike polynomials are extended to an elliptic annulus [1]_ by evaluating them at the equivalent
    polar coordinates :math:`(\rho_{eq}, \theta_{eq})` of the point (see :func:`pyzernike.cartesian_to_elliptic_annulus`):

    .. math::

        Zxy_{n}^{m}(x, y) = Z_{n}^{m}\left(\rho_{eq}(x, y), \theta_{eq}(x, y)\right)

    **Derivatives.** Let :math:`u = X / R_x` and :math:`v = Y / R_y`, where :math:`(X, Y)` are the coordinates
    in the frame of the ellipse, :math:`r = \sqrt{u^2 + v^2}` and :math:`\theta = \operatorname{atan2}(v, u)`.
    The derivatives are not computed with the chain rule in :math:`(\rho_{eq}, \theta_{eq})`, which is singular
    at the center of the ellipse (:math:`\theta` is not differentiable at :math:`r = 0`). Instead, the operators

    .. math::

        \partial_u = \cos\theta \, \partial_r - \frac{\sin\theta}{r} \partial_\theta
        \quad \text{and} \quad
        \partial_v = \sin\theta \, \partial_r + \frac{\cos\theta}{r} \partial_\theta

    are applied symbolically to the terms :math:`A(r) \cos(k\theta)` and :math:`A(r) \sin(k\theta)`. With
    :math:`P = A' + \frac{k}{r} A` and :math:`M = A' - \frac{k}{r} A`:

    .. math::

        \partial_u [A \cos k\theta] = \tfrac{1}{2} \left[ P \cos(k-1)\theta + M \cos(k+1)\theta \right],
        \quad
        \partial_v [A \cos k\theta] = \tfrac{1}{2} \left[ M \sin(k+1)\theta - P \sin(k-1)\theta \right]

    .. math::

        \partial_u [A \sin k\theta] = \tfrac{1}{2} \left[ M \sin(k+1)\theta + P \sin(k-1)\theta \right],
        \quad
        \partial_v [A \sin k\theta] = \tfrac{1}{2} \left[ P \cos(k-1)\theta - M \cos(k+1)\theta \right]

    The result has the same form, so the rules can be applied again for any order. For :math:`h = 0`, the
    negative powers of :math:`r` cancel exactly (the Cartesian derivatives of a Zernike polynomial are
    combinations of Zernike polynomials of lower order [2]_), so the result is exact at the center. For
    :math:`h > 0`, :math:`r \geq h > 0` on the elliptic annulus. Finally, :math:`(x, y) \mapsto (u, v)` is affine
    (see the section *Cartesian derivatives* of the page :doc:`../../mathematical_description/elliptic_annulus`):

    .. math::

        \partial_x = \frac{\cos\alpha}{R_x} \partial_u - \frac{\sin\alpha}{R_y} \partial_v
        \quad \text{and} \quad
        \partial_y = \frac{\sin\alpha}{R_x} \partial_u + \frac{\cos\alpha}{R_y} \partial_v

    - If :math:`|m| > n` or :math:`n - m` is odd, the polynomial is zero.
    - Where the point is outside the elliptic annulus (:math:`\rho_{eq}` outside :math:`[0, 1]`), or the
      coordinates are not finite, the output is set to ``default``.

    **Shape of the inputs and of the output.** ``n``, ``m``, ``x_derivative`` and ``y_derivative`` can
    each be a single integer or a sequence of integers (list, tuple or 1D numpy array):

    - If all of them are integers, a single polynomial is computed and returned as an array with the
      shape of ``x``.
    - If at least one of them is a sequence, the integers are repeated to the length :math:`N` of the
      sequences (which must all have the same length), and :math:`N` polynomials are computed. They are
      returned as a list of :math:`N` arrays with the shape of ``x`` (``to_list=True``), or as a single
      array of shape :math:`(N, \dots)` (``to_list=False``).

    .. seealso::

        - :func:`pyzernike.zernike_polynomial` to compute the Zernike polynomials in polar coordinates.
        - :func:`pyzernike.cartesian_to_elliptic_annulus` to compute the equivalent polar coordinates.
        - :func:`pyzernike.xy_zernike_polynomial_up_to_order` to compute all the polynomials up to an order (with the
          recurrence relation for high orders).
        - :func:`pyzernike.xy_zernike_evaluate` to evaluate a sum of polynomials from its coefficients.
        - The page :doc:`../../mathematical_description/elliptic_annulus` for the extension of the Zernike polynomials to the elliptic annulus.

    Parameters
    ----------
    x : numpy.ndarray
        The x Cartesian coordinates with any shape ``(...)``.

    y : numpy.ndarray
        The y Cartesian coordinates with the same shape as ``x``.
        If ``x`` and ``y`` are floating point arrays of the same dtype (e.g. ``numpy.float32``), this
        dtype is kept. Otherwise, both are converted to ``numpy.float64``.

    n : Union[Integral, Sequence[Integral], numpy.ndarray]
        The radial order(s) :math:`n`.

    m : Union[Integral, Sequence[Integral], numpy.ndarray]
        The azimuthal frequency(ies) :math:`m`.

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

    x_derivative : Optional[Union[Integral, Sequence[Integral], numpy.ndarray]], optional
        The non-negative order(s) of the derivative with respect to :math:`x`. Default is 0 (no derivative).

    y_derivative : Optional[Union[Integral, Sequence[Integral], numpy.ndarray]], optional
        The non-negative order(s) of the derivative with respect to :math:`y`. Default is 0 (no derivative).

    default : Union[int, float], optional
        The value set outside the elliptic annulus or where the coordinates are not finite. Integers and floating
        point numbers (Python or numpy, including ``numpy.nan`` and ``numpy.inf``) are accepted and converted to
        float. Booleans are not accepted. Default is ``numpy.nan``.

    precompute : bool, optional
        If True, the powers of :math:`\rho_{eq}` and :math:`1/r` and the cosine and sine terms are cached and
        shared between the polynomials, which is faster when several polynomials are computed at once. If
        False, they are recomputed for each polynomial, which uses less memory. Default is True.

    normalize : bool, optional
        If True, the polynomials (and their derivatives) are multiplied by the normalization factors of Noll
        :math:`N_{n}^{m} = \sqrt{(2 - \delta_{m0})(n + 1)}` (see :func:`pyzernike.zernike_normalization`). For :math:`h = 0`,
        the normalized polynomials are orthonormal on the ellipse for the inner product averaged over its area. The
        value ``default`` is not normalized. Default is False.

    to_list : bool, optional
        Only used when several polynomials are computed. If True, the output is a list of arrays.
        If False, the output is a single array of shape :math:`(N, \dots)`. Default is True.

    None is accepted for all the parameters of the ellipse and for the derivative orders, and means the default value.

    Returns
    -------
    Union[numpy.ndarray, List[numpy.ndarray]]
        - If ``n``, ``m``, ``x_derivative`` and ``y_derivative`` are integers: an array with the shape of ``x``.
        - Otherwise, if ``to_list`` is True: a list of :math:`N` arrays with the shape of ``x``.
        - Otherwise: an array of shape :math:`(N, \dots)`, where ``output[i]`` corresponds to
          ``(n[i], m[i], x_derivative[i], y_derivative[i])``.

        The dtype is the common floating point dtype of ``x`` and ``y`` (``numpy.float64`` by default).

    Raises
    ------
    TypeError
        If ``x`` or ``y`` can not be converted to a floating point array, if a parameter of the ellipse
        is not a real number, if ``default`` is not an integer or a floating point number, if ``n``, ``m``, ``x_derivative`` or ``y_derivative`` is not
        an integer or a sequence of integers, or if ``precompute``, ``normalize`` or ``to_list`` is not a boolean.

    ValueError
        If ``x`` and ``y`` do not have the same shape, if a parameter of the ellipse is not finite, if
        ``Rx`` or ``Ry`` is not strictly positive, if ``h`` is not in :math:`[0, 1)`, if the sequences do
        not have the same length, or if a derivative order is negative.

    References
    ----------
    .. [1] R. Navarro, J. L. López, J. A. Díaz and E. Pérez Sinusía, "Generalization of Zernike polynomials for
       regular portions of circles and ellipses," Optics Express 22(18), 21263-21279 (2014).
       https://doi.org/10.1364/OE.22.021263

    .. [2] R. J. Noll, "Zernike polynomials and atmospheric turbulence," Journal of the Optical Society
       of America 66(3), 207-211 (1976). https://doi.org/10.1364/JOSA.66.000207

    Examples
    --------
    Compute the defocus :math:`Z_{2}^{0}` on a circle of radius 10 centered at :math:`(1, 1)`:

    .. code-block:: python

        import numpy
        from pyzernike import xy_zernike_polynomial

        X, Y = numpy.meshgrid(numpy.linspace(-10, 10, 100), numpy.linspace(-10, 10, 100))
        Z20 = xy_zernike_polynomial(X, Y, n=2, m=0, Rx=10, Ry=10, x0=1.0, y0=1.0)  # array of shape (100, 100)

    Compute its gradient and its Hessian matrix (``n`` and ``m`` are repeated):

    .. code-block:: python

        dZ_dx, dZ_dy = xy_zernike_polynomial(X, Y, n=2, m=0, Rx=10, Ry=10, x0=1.0, y0=1.0, x_derivative=[1, 0], y_derivative=[0, 1])
        d2Z_dx2, d2Z_dy2, d2Z_dxdy = xy_zernike_polynomial(X, Y, n=2, m=0, Rx=10, Ry=10, x0=1.0, y0=1.0, x_derivative=[2, 0, 1], y_derivative=[0, 2, 1])

    """
    x, y = _as_common_floating_arrays(x=x, y=y)
    Rx, Ry, x0, y0, alpha, h = _as_ellipse_parameters(Rx, Ry, x0, y0, alpha, h)
    default = _as_default_value(default)
    precompute = _check_boolean(precompute, "precompute")
    normalize = _check_boolean(normalize, "normalize")
    to_list = _check_boolean(to_list, "to_list")
    indices, single = _broadcast_indices(
        integers={"n": n, "m": m},
        orders={"x_derivative": x_derivative, "y_derivative": y_derivative},
    )

    float_type = x.dtype.type
    rho_list, theta_list = _core_cartesian_to_elliptic_annulus(
        x=x,
        y=y,
        Rx=float_type(Rx),
        Ry=float_type(Ry),
        x0=float_type(x0),
        y0=float_type(y0),
        alpha=float_type(alpha),
        h=float_type(h),
        x_derivative=[0],
        y_derivative=[0],
    )
    rho_eq, theta_eq = rho_list[0], theta_list[0]

    valid_mask = numpy.isfinite(rho_eq) & numpy.isfinite(theta_eq) & (rho_eq >= 0) & (rho_eq <= 1)

    output = numpy.full((len(indices["n"]),) + x.shape, default, dtype=x.dtype)
    output[:, valid_mask] = _core_xy_polynomials(
        rho=rho_eq[valid_mask],
        theta=theta_eq[valid_mask],
        n=indices["n"],
        m=indices["m"],
        x_derivative=indices["x_derivative"],
        y_derivative=indices["y_derivative"],
        Rx=Rx,
        Ry=Ry,
        alpha=alpha,
        h=h,
        precompute=precompute,
        normalize=normalize,
    )

    return _format_output(output, single, to_list)