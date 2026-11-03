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
    _as_order,
    _broadcast_indices,
    _check_boolean,
)
from ..core.core_cartesian_to_elliptic_annulus import _core_cartesian_to_elliptic_annulus
from ..core.core_xy_polynomial import _core_xy_polynomials
from ..core.core_xy_recursive import _core_xy_recursive_up_to_order
from ..core.core_indexing import _core_index_to_order, _core_polynomials_count


def xy_zernike_polynomial_up_to_order(
    x: numpy.ndarray,
    y: numpy.ndarray,
    order: Integral,
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
    use_recursive: bool = False,
    normalize: bool = False,
    to_list: bool = True,
) -> Union[numpy.ndarray, List[List[numpy.ndarray]]]:
    r"""
    Compute all the Zernike polynomials up to a radial order :math:`N` on an elliptic annulus for Cartesian
    coordinates :math:`(x, y)`, and their derivatives of any order with respect to :math:`x` and :math:`y`.

    .. figure:: ../../../pyzernike/resources/elliptic_annulus_domain.png
        :width: 400px
        :align: center

        The parameters of the elliptic annulus domain.

    The Zernike polynomials are extended to an elliptic annulus [1]_ by evaluating them at the equivalent
    polar coordinates :math:`(\rho_{eq}, \theta_{eq})` of the point (see :func:`pyzernike.cartesian_to_elliptic_annulus`).
    All the :math:`P = \frac{(N+1)(N+2)}{2}` polynomials with :math:`0 \leq n \leq N` are computed, in the
    OSA/ANSI ordering: the polynomial :math:`Z_{n}^{m}` is at the index :math:`j = \frac{n(n+2) + m}{2}`
    (see :func:`pyzernike.zernike_index_to_order`).

    The derivatives with respect to :math:`u = X / R_x` and :math:`v = Y / R_y` are built symbolically with exact
    coefficients, without the chain rule in :math:`(\rho_{eq}, \theta_{eq})` (see :func:`pyzernike.xy_zernike_polynomial`).
    Any derivative order is available, and the derivatives are exact at the center of the ellipse. Two methods are
    available to apply these coefficients:

    - ``use_recursive=False``: the explicit sums with exact integer coefficients, evaluated with the Horner scheme.
      The cancellation between the large alternating terms of the
      sums limits the accuracy for high orders (the error grows quickly beyond :math:`n \approx 20`).
    - ``use_recursive=True``: the coefficients are applied to values computed with the recurrence relation of
      Shakibaei and Paramesran [2]_. For :math:`h = 0`,
      the derivatives are linear combinations of Zernike polynomials of lower orders [3]_; for :math:`h > 0`, they are
      combinations of the radial derivatives :math:`\frac{d^a R_{n}^{|m|}}{d\rho^a}(\rho_{eq})`. The accuracy stays
      close to the machine precision at high orders. This method is recommended for high orders.

    Both methods are described in the sections *Cartesian derivatives* and *Recursive mode* of the page
    :doc:`../../mathematical_description/elliptic_annulus`.

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
        - :func:`pyzernike.xy_zernike_evaluate` to evaluate a sum of polynomials from its coefficients, without
          storing all the polynomials.
        - The page :doc:`../../mathematical_description/elliptic_annulus` for the extension of the Zernike polynomials to the elliptic annulus.

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

    default : Union[int, float], optional
        The value set outside the elliptic annulus or where the coordinates are not finite. Integers and floating
        point numbers (Python or numpy, including ``numpy.nan`` and ``numpy.inf``) are accepted and converted to
        float. Booleans are not accepted. Default is ``numpy.nan``.

    precompute : bool, optional
        Only used if ``use_recursive`` is False. If True, the powers of :math:`\rho_{eq}` and :math:`1/r` and the
        cosine and sine terms are cached and shared between the polynomials. If False, they are recomputed for each
        polynomial, which uses less memory. Default is True.

    use_recursive : bool, optional
        If True, the polynomials and their derivatives are computed with the recurrence relation instead of the
        explicit sums (accurate at high orders). Default is False.

    normalize : bool, optional
        If True, the polynomials (and their derivatives) are multiplied by the normalization factors of Noll
        :math:`N_{n}^{m} = \sqrt{(2 - \delta_{m0})(n + 1)}` (see :func:`pyzernike.zernike_normalization`). For :math:`h = 0`,
        the normalized polynomials are orthonormal on the ellipse for the inner product averaged over its area. The
        value ``default`` is not normalized. Default is False.

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
        parameter of the ellipse is not a real number, if ``default`` is not an integer or a floating point
        number, if ``x_derivative`` or ``y_derivative`` is not an integer or a sequence of integers, or if
        ``precompute``, ``use_recursive``, ``normalize`` or ``to_list`` is not a boolean.

    ValueError
        If ``x`` and ``y`` do not have the same shape, if ``order`` is negative, if a parameter of the ellipse
        is not finite, if ``Rx`` or ``Ry`` is not strictly positive, if ``h`` is not in :math:`[0, 1)`, if the
        sequences do not have the same length, or if a derivative order is negative.

    References
    ----------
    .. [1] R. Navarro, J. L. López, J. A. Díaz and E. Pérez Sinusía, "Generalization of Zernike polynomials for
       regular portions of circles and ellipses," Optics Express 22(18), 21263-21279 (2014).
       https://doi.org/10.1364/OE.22.021263

    .. [2] B. H. Shakibaei and R. Paramesran, "Recursive formula to compute Zernike radial polynomials,"
       Optics Letters 38(14), 2487-2489 (2013). https://doi.org/10.1364/OL.38.002487

    .. [3] R. J. Noll, "Zernike polynomials and atmospheric turbulence," Journal of the Optical Society
       of America 66(3), 207-211 (1976). https://doi.org/10.1364/JOSA.66.000207

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

    Compute all the polynomials up to the order 40 and their gradient with the recurrence relation:

    .. code-block:: python

        Z = xy_zernike_polynomial_up_to_order(
            X, Y, order=40, Rx=numpy.sqrt(2), Ry=numpy.sqrt(2),
            x_derivative=[0, 1, 0], y_derivative=[0, 0, 1], use_recursive=True, to_list=False,
        )
        # Z.shape == (3, 861, 100, 100)

    """
    x, y = _as_common_floating_arrays(x=x, y=y)
    order = _as_order(order)
    Rx, Ry, x0, y0, alpha, h = _as_ellipse_parameters(Rx, Ry, x0, y0, alpha, h)
    default = _as_default_value(default)
    precompute = _check_boolean(precompute, "precompute")
    use_recursive = _check_boolean(use_recursive, "use_recursive")
    normalize = _check_boolean(normalize, "normalize")
    to_list = _check_boolean(to_list, "to_list")
    derivatives, _ = _broadcast_indices(
        integers={},
        orders={"x_derivative": x_derivative, "y_derivative": y_derivative},
    )
    x_derivative = derivatives["x_derivative"]
    y_derivative = derivatives["y_derivative"]

    n_derivatives = len(x_derivative)
    n_polynomials = _core_polynomials_count(order)

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
    all_valid = bool(valid_mask.all())
    rho_valid = rho_eq.ravel() if all_valid else rho_eq[valid_mask]
    theta_valid = theta_eq.ravel() if all_valid else theta_eq[valid_mask]

    if use_recursive:
        values = _core_xy_recursive_up_to_order(
            rho=rho_valid,
            theta=theta_valid,
            order=order,
            x_derivative=x_derivative,
            y_derivative=y_derivative,
            Rx=Rx,
            Ry=Ry,
            alpha=alpha,
            h=h,
            normalize=normalize,
        )
    else:
        n, m = _core_index_to_order(list(range(n_polynomials)), indexing="OSA")
        values = _core_xy_polynomials(
            rho=rho_valid,
            theta=theta_valid,
            n=n * n_derivatives,
            m=m * n_derivatives,
            x_derivative=[dx for dx in x_derivative for _ in range(n_polynomials)],
            y_derivative=[dy for dy in y_derivative for _ in range(n_polynomials)],
            Rx=Rx,
            Ry=Ry,
            alpha=alpha,
            h=h,
            precompute=precompute,
            normalize=normalize,
        ).reshape((n_derivatives, n_polynomials, rho_valid.size))

    if all_valid:
        output = values.reshape((n_derivatives, n_polynomials) + x.shape)
    else:
        output = numpy.full((n_derivatives, n_polynomials) + x.shape, default, dtype=x.dtype)
        output[:, :, valid_mask] = values

    if to_list:
        return [list(output[d]) for d in range(n_derivatives)]
    return output