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

from .core.parsing import _as_common_floating_arrays, _as_ellipse_parameters, _broadcast_integers
from .core.cartesian_to_elliptic_annulus import _cartesian_to_elliptic_annulus
from .core.zernike_recursive import _zernike_recursive_up_to_order
from .tools import zernike_index_to_order, zernike_polynomials_count
from .xy_zernike_polynomial import COORDINATE_DERIVATIVES, POLAR_DERIVATIVES, _chain_rule, xy_zernike_polynomial


def xy_zernike_polynomial_up_to_order(
    x: numpy.ndarray,
    y: numpy.ndarray,
    order: Integral,
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
    use_recursive: bool = True,
    to_list: bool = True,
) -> Union[numpy.ndarray, List[List[numpy.ndarray]]]:
    r"""
    Compute all the Zernike polynomials up to a radial order :math:`N` on an elliptic annulus for Cartesian
    coordinates :math:`(x, y)`, and their first and second derivatives with respect to :math:`x` and :math:`y`.

    .. figure:: ../../../pyzernike/resources/elliptic_annulus_domain.png
        :width: 400px
        :align: center

        The parameters of the elliptic annulus domain.

    The Zernike polynomials are extended to an elliptic annulus [1]_ by evaluating them at the equivalent
    polar coordinates :math:`(\rho_{eq}, \theta_{eq})` of the point (see :func:`pyzernike.cartesian_to_elliptic_annulus`),
    and their derivatives with respect to :math:`x` and :math:`y` are obtained with the chain rule (see
    :func:`pyzernike.xy_zernike_polynomial`). All the :math:`P = \frac{(N+1)(N+2)}{2}` polynomials with
    :math:`0 \leq n \leq N` are computed, in the OSA/ANSI ordering: the polynomial :math:`Z_{n}^{m}` is at
    the index :math:`j = \frac{n(n+2) + m}{2}` (see :func:`pyzernike.zernike_index_to_order`).

    Two methods are available to compute the polar derivatives of the polynomials:

    - ``use_recursive=True``: the three-term recurrence relation of Shakibaei and Paramesran [2]_ (see
      :func:`pyzernike.core._zernike_recursive_up_to_order`), called once for all the polynomials and all
      the polar derivatives required by the chain rule. The cost is :math:`O(N^2)` operations per point, and
      the accuracy stays close to the machine precision at any order.
    - ``use_recursive=False``: the explicit sum with exact integer coefficients, through
      :func:`pyzernike.xy_zernike_polynomial`. The cost is :math:`O(N^3)` operations per point, and the
      cancellation between the large alternating terms of the sum limits the accuracy for high orders
      (the absolute error in double precision grows quickly beyond :math:`n \approx 20`).

    The available derivatives ``(x_derivative, y_derivative)`` are ``(0, 0)``, ``(1, 0)``, ``(0, 1)``
    (gradient), and ``(2, 0)``, ``(0, 2)``, ``(1, 1)`` (Hessian matrix).

    Where the point is outside the elliptic annulus (:math:`\rho_{eq}` outside :math:`[0, 1]`), or the
    coordinates are not finite, the output is set to ``default``.

    **Shape of the inputs and of the output.** ``x_derivative`` and ``y_derivative`` can each be a single
    integer or a sequence of integers (list, tuple or 1D numpy array). The integer is repeated to the length
    :math:`D` of the sequence (the sequences must have the same length), and each pair
    ``(x_derivative[d], y_derivative[d])`` gives one set of :math:`P` polynomials. The output always has a
    first dimension of length :math:`D` (:math:`D = 1` if both are integers):

    - ``to_list=True``: a list of :math:`D` lists of :math:`P` arrays with the shape of ``x``.
    - ``to_list=False``: a single array of shape :math:`(D, P, \dots)`.

    .. seealso::

        - :func:`pyzernike.xy_zernike_polynomial` to compute given polynomials :math:`(n, m)` on the elliptic annulus.
        - :func:`pyzernike.zernike_polynomial_up_to_order` to compute all the polynomials up to an order in polar coordinates.
        - :func:`pyzernike.zernike_index_to_order` to convert the OSA/ANSI indices to orders :math:`(n, m)`.
        - The page :doc:`../../mathematical_description` for the extension of the Zernike polynomials to the elliptic annulus.

    Parameters
    ----------
    x : numpy.ndarray
        The x Cartesian coordinates with any shape ``(...)``.

    y : numpy.ndarray
        The y Cartesian coordinates with the same shape as ``x``.
        If ``x`` and ``y`` are floating point arrays of the same dtype (e.g. ``numpy.float32``), this
        dtype is kept. Otherwise, both are converted to ``numpy.float64``.

    order : Integral
        The maximum radial order :math:`N \geq 0`.

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
        Only used if ``use_recursive`` is False. If True, the powers of :math:`\rho` and the cosine and sine
        terms are cached and shared between the polynomials. If False, they are recomputed for each
        polynomial, which uses less memory. Default is True.

    use_recursive : bool, optional
        If True, the polynomials are computed with the recurrence relation instead of the explicit sum
        (faster and more accurate for high orders). Default is True.

    to_list : bool, optional
        If True, the output is a list of lists of arrays. If False, the output is a single array of shape
        :math:`(D, P, \dots)`. Default is True.

    None is accepted for all the parameters of the ellipse and for the derivative orders, and means the default value.

    Returns
    -------
    Union[List[List[numpy.ndarray]], numpy.ndarray]
        The polynomials, where ``output[d][j]`` (``to_list=True``) or ``output[d, j]`` (``to_list=False``) is the
        polynomial of OSA/ANSI index ``j`` for the pair ``(x_derivative[d], y_derivative[d])``, with the shape
        of ``x``. The dtype is the common floating point dtype of ``x`` and ``y`` (``numpy.float64`` by default).

    Raises
    ------
    TypeError
        If ``x`` or ``y`` can not be converted to a floating point array, if ``order`` is not an integer, if a
        parameter of the ellipse or ``default`` is not a real number, if ``x_derivative`` or ``y_derivative``
        is not an integer or a sequence of integers, or if ``precompute``, ``use_recursive`` or ``to_list``
        is not a boolean.

    ValueError
        If ``x`` and ``y`` do not have the same shape, if ``order`` is negative, if ``Rx`` or ``Ry`` is not
        strictly positive, if ``h`` is not in :math:`[0, 1)`, if the sequences do not have the same length,
        or if a derivative ``(x_derivative, y_derivative)`` is not available.

    References
    ----------
    .. [1] R. Navarro et al., "Generalization of Zernike polynomials for regular portions of circles
       and ellipses," Optics Express 22(18), 21263-21279 (2014). https://doi.org/10.1364/OE.22.021263

    .. [2] B. H. Shakibaei and R. Paramesran, "Recursive formula to compute Zernike radial polynomials,"
       Optics Letters 38(14), 2487 (2013). https://doi.org/10.1364/OL.38.002487

    Examples
    --------
    Compute all the Zernike polynomials up to the order 3 on a circle of radius :math:`\sqrt{2}` centered at the origin:

    .. code-block:: python

        import numpy
        from pyzernike import xy_zernike_polynomial_up_to_order, zernike_index_to_order

        X, Y = numpy.meshgrid(numpy.linspace(-1, 1, 100), numpy.linspace(-1, 1, 100))
        polynomials = xy_zernike_polynomial_up_to_order(X, Y, order=3, Rx=numpy.sqrt(2), Ry=numpy.sqrt(2))[0]  # list of 10 arrays

        n, m = zernike_index_to_order(range(len(polynomials)))
        for n_j, m_j, Z in zip(n, m, polynomials):
            print(f"Z_{n_j}^{m_j}: max = {numpy.nanmax(Z):.3f}")

    Compute all the polynomials up to the order 10, their gradient and their Hessian matrix as a single array:

    .. code-block:: python

        Z = xy_zernike_polynomial_up_to_order(
            X, Y, order=10, Rx=numpy.sqrt(2), Ry=numpy.sqrt(2),
            x_derivative=[0, 1, 0, 2, 0, 1], y_derivative=[0, 0, 1, 0, 2, 1], to_list=False,
        )
        # Z.shape == (6, 66, 100, 100)

    """
    x, y = _as_common_floating_arrays(x=x, y=y)
    if isinstance(order, bool) or not isinstance(order, Integral):
        raise TypeError("order must be an integer.")
    if order < 0:
        raise ValueError("order must be a non-negative integer.")
    Rx, Ry, x0, y0, alpha, h = _as_ellipse_parameters(Rx, Ry, x0, y0, alpha, h)
    if not isinstance(default, Real):
        raise TypeError("default must be a real number.")
    for name, value in (("precompute", precompute), ("use_recursive", use_recursive), ("to_list", to_list)):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a boolean.")

    if x_derivative is None:
        x_derivative = 0
    if y_derivative is None:
        y_derivative = 0

    derivatives, _ = _broadcast_integers(x_derivative=x_derivative, y_derivative=y_derivative)
    derivatives = list(zip(derivatives["x_derivative"], derivatives["y_derivative"]))
    for derivative in derivatives:
        if derivative not in POLAR_DERIVATIVES:
            raise ValueError(
                f"The derivative (x_derivative, y_derivative) = {derivative} is not available. "
                "The available derivatives are (0, 0), (1, 0), (0, 1), (2, 0), (0, 2) and (1, 1)."
            )

    order = int(order)
    n_derivatives = len(derivatives)
    n_polynomials = zernike_polynomials_count(order)

    if not use_recursive:
        n, m = zernike_index_to_order(list(range(n_polynomials)))
        output = xy_zernike_polynomial(
            x, y, n * n_derivatives, m * n_derivatives,
            Rx, Ry, x0, y0, alpha, h,
            [dx for dx, _ in derivatives for _ in range(n_polynomials)],
            [dy for _, dy in derivatives for _ in range(n_polynomials)],
            default,
            precompute=precompute,
            to_list=False,
        ).reshape((n_derivatives, n_polynomials) + x.shape)

    else:
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
        all_valid = bool(valid_mask.all())
        rho_valid = {pair: value.ravel() if all_valid else value[valid_mask] for pair, value in rho_eq.items()}
        theta_valid = {pair: value.ravel() if all_valid else value[valid_mask] for pair, value in theta_eq.items()}

        polar_derivatives = sorted({pair for d in derivatives for pair in POLAR_DERIVATIVES[d]})
        polar_values = _zernike_recursive_up_to_order(
            rho_valid[(0, 0)],
            theta_valid[(0, 0)],
            order,
            [a for a, _ in polar_derivatives],
            [b for _, b in polar_derivatives],
        )
        polar = dict(zip(polar_derivatives, polar_values))

        if all_valid:
            output = numpy.empty((n_derivatives, n_polynomials) + x.shape, dtype=x.dtype)
            flat_output = output.reshape((n_derivatives, n_polynomials, -1))
            for index, derivative in enumerate(derivatives):
                flat_output[index] = _chain_rule(derivative, polar, rho_valid, theta_valid)
        else:
            output = numpy.full((n_derivatives, n_polynomials) + x.shape, default, dtype=x.dtype)
            for index, derivative in enumerate(derivatives):
                output[index][:, valid_mask] = _chain_rule(derivative, polar, rho_valid, theta_valid)

    if to_list:
        return [list(output[d]) for d in range(n_derivatives)]
    return output