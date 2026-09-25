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
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy

from .core.parsing import _as_common_floating_arrays, _as_ellipse_parameters, _broadcast_integers
from .core.core_polynomial import _core_polynomials
from .core.cartesian_to_elliptic_annulus import _cartesian_to_elliptic_annulus

Pair = Tuple[int, int]

# Cartesian derivative (dx, dy) -> polar derivatives (a, b) of Z required by the chain rule
POLAR_DERIVATIVES: Dict[Pair, List[Pair]] = {
    (0, 0): [(0, 0)],
    (1, 0): [(1, 0), (0, 1)],
    (0, 1): [(1, 0), (0, 1)],
    (2, 0): [(1, 0), (0, 1), (2, 0), (0, 2), (1, 1)],
    (0, 2): [(1, 0), (0, 1), (2, 0), (0, 2), (1, 1)],
    (1, 1): [(1, 0), (0, 1), (2, 0), (0, 2), (1, 1)],
}

# Cartesian derivative (dx, dy) -> Cartesian derivatives of (rho_eq, theta_eq) required by the chain rule
COORDINATE_DERIVATIVES: Dict[Pair, List[Pair]] = {
    (0, 0): [],
    (1, 0): [(1, 0)],
    (0, 1): [(0, 1)],
    (2, 0): [(1, 0), (2, 0)],
    (0, 2): [(0, 1), (0, 2)],
    (1, 1): [(1, 0), (0, 1), (1, 1)],
}


def _chain_rule(
    derivative: Pair,
    Z: Dict[Pair, numpy.ndarray],
    rho: Dict[Pair, numpy.ndarray],
    theta: Dict[Pair, numpy.ndarray],
) -> numpy.ndarray:
    r"""
    Assemble a Cartesian derivative of :math:`Z(\rho_{eq}(x, y), \theta_{eq}(x, y))` with the chain rule.

    For :math:`z, w \in \{x, y\}`:

    .. math::

        \frac{\partial Z}{\partial z} = Z_{\rho} \rho_{z} + Z_{\theta} \theta_{z}

    .. math::

        \frac{\partial^2 Z}{\partial z \partial w} = Z_{\rho\rho} \rho_{z} \rho_{w}
        + Z_{\rho\theta} \left( \rho_{z} \theta_{w} + \theta_{z} \rho_{w} \right)
        + Z_{\theta\theta} \theta_{z} \theta_{w}
        + Z_{\rho} \rho_{zw} + Z_{\theta} \theta_{zw}

    Parameters
    ----------
    derivative : Tuple[int, int]
        The Cartesian derivative orders ``(dx, dy)``, with :math:`dx + dy \leq 2`.

    Z : Dict[Tuple[int, int], numpy.ndarray]
        The polar derivatives of the polynomial, indexed by ``(a, b)`` (see ``POLAR_DERIVATIVES``).

    rho, theta : Dict[Tuple[int, int], numpy.ndarray]
        The Cartesian derivatives of :math:`\rho_{eq}` and :math:`\theta_{eq}`, indexed by ``(dx, dy)``
        (see ``COORDINATE_DERIVATIVES``).

    Returns
    -------
    numpy.ndarray
        The Cartesian derivative of the polynomial.

    """
    if derivative == (0, 0):
        return Z[(0, 0)]

    if sum(derivative) == 1:
        return Z[(1, 0)] * rho[derivative] + Z[(0, 1)] * theta[derivative]

    z, w = {(2, 0): ((1, 0), (1, 0)), (0, 2): ((0, 1), (0, 1)), (1, 1): ((1, 0), (0, 1))}[derivative]
    return (
        Z[(2, 0)] * rho[z] * rho[w]
        + Z[(1, 1)] * (rho[z] * theta[w] + theta[z] * rho[w])
        + Z[(0, 2)] * theta[z] * theta[w]
        + Z[(1, 0)] * rho[derivative]
        + Z[(0, 1)] * theta[derivative]
    )


def xy_zernike_polynomial(
    x: numpy.ndarray,
    y: numpy.ndarray,
    n: Union[Integral, Sequence[Integral], numpy.ndarray],
    m: Union[Integral, Sequence[Integral], numpy.ndarray],
    Rx: Optional[Real] = 1.0,
    Ry: Optional[Real] = 1.0,
    x0: Optional[Real] = 0.0,
    y0: Optional[Real] = 0.0,
    alpha: Optional[Real] = 0.0,
    h: Optional[Real] = 0.0,
    x_derivative: Optional[Union[Integral, Sequence[Integral], numpy.ndarray]] = 0,
    y_derivative: Optional[Union[Integral, Sequence[Integral], numpy.ndarray]] = 0,
    default: Real = numpy.nan,
    *,
    precompute: bool = True,
    to_list: bool = True,
) -> Union[numpy.ndarray, List[numpy.ndarray]]:
    r"""
    Compute the Zernike polynomials on an elliptic annulus for Cartesian coordinates :math:`(x, y)`, and
    their first and second derivatives with respect to :math:`x` and :math:`y`.

    .. figure:: ../../../pyzernike/resources/elliptic_annulus_domain.png
        :width: 400px
        :align: center

        The parameters of the elliptic annulus domain.

    The Zernike polynomials are extended to an elliptic annulus [1]_ by evaluating them at the equivalent
    polar coordinates :math:`(\rho_{eq}, \theta_{eq})` of the point (see :func:`pyzernike.cartesian_to_elliptic_annulus`):

    .. math::

        Zxy_{n}^{m}(x, y) = Z_{n}^{m}\left(\rho_{eq}(x, y), \theta_{eq}(x, y)\right)

    The derivatives with respect to :math:`x` and :math:`y` are obtained with the chain rule. For
    :math:`z, w \in \{x, y\}`:

    .. math::

        \frac{\partial Zxy_{n}^{m}}{\partial z} = \frac{\partial Z_{n}^{m}}{\partial \rho} \frac{\partial \rho_{eq}}{\partial z} + \frac{\partial Z_{n}^{m}}{\partial \theta} \frac{\partial \theta_{eq}}{\partial z}

    .. math::

        \frac{\partial^2 Zxy_{n}^{m}}{\partial z \partial w} =
        \frac{\partial^2 Z_{n}^{m}}{\partial \rho^2} \frac{\partial \rho_{eq}}{\partial z} \frac{\partial \rho_{eq}}{\partial w}
        + \frac{\partial^2 Z_{n}^{m}}{\partial \rho \partial \theta} \left( \frac{\partial \rho_{eq}}{\partial z} \frac{\partial \theta_{eq}}{\partial w} + \frac{\partial \theta_{eq}}{\partial z} \frac{\partial \rho_{eq}}{\partial w} \right)
        + \frac{\partial^2 Z_{n}^{m}}{\partial \theta^2} \frac{\partial \theta_{eq}}{\partial z} \frac{\partial \theta_{eq}}{\partial w}
        + \frac{\partial Z_{n}^{m}}{\partial \rho} \frac{\partial^2 \rho_{eq}}{\partial z \partial w}
        + \frac{\partial Z_{n}^{m}}{\partial \theta} \frac{\partial^2 \theta_{eq}}{\partial z \partial w}

    The available derivatives ``(x_derivative, y_derivative)`` are ``(0, 0)``, ``(1, 0)``, ``(0, 1)``
    (gradient), and ``(2, 0)``, ``(0, 2)``, ``(1, 1)`` (Hessian matrix). For higher orders, use
    :func:`pyzernike.cartesian_to_elliptic_annulus` and :func:`pyzernike.zernike_polynomial` with the
    Faà di Bruno's formula.

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
        - The page :doc:`../../mathematical_description` for the extension of the Zernike polynomials to the elliptic annulus.

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

    default : Real, optional
        The value set outside the elliptic annulus or where the coordinates are not finite. Default is ``numpy.nan``.

    precompute : bool, optional
        If True, the powers of :math:`\rho` and the cosine and sine terms are cached and shared between the
        polynomials, which is faster when several polynomials are computed at once. If False, they are
        recomputed for each polynomial, which uses less memory. Default is True.

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
        or ``default`` is not a real number, if ``n``, ``m``, ``x_derivative`` or ``y_derivative`` is not
        an integer or a sequence of integers, or if ``precompute`` or ``to_list`` is not a boolean.

    ValueError
        If ``x`` and ``y`` do not have the same shape, if ``Rx`` or ``Ry`` is not strictly positive, if
        ``h`` is not in :math:`[0, 1)`, if the sequences do not have the same length, or if a derivative
        ``(x_derivative, y_derivative)`` is not available.

    References
    ----------
    .. [1] R. Navarro et al., "Generalization of Zernike polynomials for regular portions of circles
       and ellipses," Optics Express 22(18), 21263-21279 (2014). https://doi.org/10.1364/OE.22.021263

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
    if not isinstance(default, Real):
        raise TypeError("default must be a real number.")
    if not isinstance(precompute, bool):
        raise TypeError("precompute must be a boolean.")
    if not isinstance(to_list, bool):
        raise TypeError("to_list must be a boolean.")

    if x_derivative is None:
        x_derivative = 0
    if y_derivative is None:
        y_derivative = 0

    indices, single = _broadcast_integers(n=n, m=m, x_derivative=x_derivative, y_derivative=y_derivative)
    derivatives = list(zip(indices["x_derivative"], indices["y_derivative"]))
    for derivative in derivatives:
        if derivative not in POLAR_DERIVATIVES:
            raise ValueError(
                f"The derivative (x_derivative, y_derivative) = {derivative} is not available. "
                "The available derivatives are (0, 0), (1, 0), (0, 1), (2, 0), (0, 2) and (1, 1)."
            )

    coordinate_derivatives = sorted({(0, 0)} | {pair for d in derivatives for pair in COORDINATE_DERIVATIVES[d]})
    float_type = x.dtype.type
    rho_list, theta_list = _cartesian_to_elliptic_annulus(
        x, y,
        float_type(Rx), float_type(Ry), float_type(x0), float_type(y0), float_type(alpha), float_type(h),
        [dx for dx, _ in coordinate_derivatives],
        [dy for _, dy in coordinate_derivatives],
    )
    rho_eq = dict(zip(coordinate_derivatives, rho_list))
    theta_eq = dict(zip(coordinate_derivatives, theta_list))

    valid_mask = numpy.isfinite(rho_eq[(0, 0)]) & numpy.isfinite(theta_eq[(0, 0)]) & (rho_eq[(0, 0)] >= 0) & (rho_eq[(0, 0)] <= 1)
    rho_valid = {pair: value[valid_mask] for pair, value in rho_eq.items()}
    theta_valid = {pair: value[valid_mask] for pair, value in theta_eq.items()}

    polar_keys = sorted({(n_i, m_i, a, b) for n_i, m_i, d in zip(indices["n"], indices["m"], derivatives) for a, b in POLAR_DERIVATIVES[d]})
    polar_values = _core_polynomials(
        rho_valid[(0, 0)],
        theta_valid[(0, 0)],
        [key[0] for key in polar_keys],
        [key[1] for key in polar_keys],
        [key[2] for key in polar_keys],
        [key[3] for key in polar_keys],
        False,
        precompute,
    )
    polar = dict(zip(polar_keys, polar_values))

    output = numpy.full((len(derivatives),) + x.shape, default, dtype=x.dtype)
    for index, (n_i, m_i, derivative) in enumerate(zip(indices["n"], indices["m"], derivatives)):
        Z = {(a, b): polar[(n_i, m_i, a, b)] for a, b in POLAR_DERIVATIVES[derivative]}
        output[index, valid_mask] = _chain_rule(derivative, Z, rho_valid, theta_valid)

    if single:
        return output[0]
    if to_list:
        return list(output)
    return output