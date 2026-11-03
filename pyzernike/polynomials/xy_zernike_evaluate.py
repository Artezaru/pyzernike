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
    _as_osa_coefficients,
    _broadcast_indices,
    _check_boolean,
    _format_output,
)
from ..core.core_cartesian_to_elliptic_annulus import _core_cartesian_to_elliptic_annulus
from ..core.core_evaluate import _core_xy_evaluate


def xy_zernike_evaluate(
    coefficients: Union[Sequence[float], numpy.ndarray],
    x: numpy.ndarray,
    y: numpy.ndarray,
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
) -> Union[numpy.ndarray, List[numpy.ndarray]]:
    r"""
    Evaluate a sum of Zernike polynomials on an elliptic annulus :math:`f = \sum_j c_j Zxy_j` and its derivatives
    with respect to the Cartesian coordinates :math:`x` and :math:`y`.

    .. figure:: ../../../pyzernike/resources/elliptic_annulus_domain.png
        :width: 400px
        :align: center

        The parameters of the elliptic annulus domain.

    .. math::

        \frac{\partial^{a+b} f}{\partial x^{a} \partial y^{b}}(x, y)
        = \sum_{j=0}^{P-1} c_j \frac{\partial^{a+b} Zxy_j}{\partial x^{a} \partial y^{b}}(x, y)

    where :math:`Zxy_j` is the Zernike polynomial of OSA/ANSI index :math:`j` on the elliptic annulus (see
    :func:`pyzernike.xy_zernike_polynomial`). The derivatives of any order are exact, also at the center of the
    ellipse. The coefficients must contain all the polynomials up to a maximum radial order :math:`N`, so their
    number must be :math:`P = \frac{(N+1)(N+2)}{2}` (1, 3, 6, 10, 15, ...): pad them with zeros if needed.

    The polynomials are computed by chunks of points, so that the :math:`P` arrays of the polynomials are never
    stored for all the points at once.

    Where the point is outside the elliptic annulus (:math:`\rho_{eq}` outside :math:`[0, 1]`), or the coordinates
    are not finite, the output is set to ``default``.

    **Shape of the inputs and of the output.** ``x_derivative`` and ``y_derivative`` can each be a single integer
    or a sequence of integers (list, tuple or 1D numpy array):

    - If both are integers, the output is an array with the shape of ``x``.
    - If at least one of them is a sequence, the integer is repeated to the length :math:`D` of the sequence (the
      sequences must have the same length), and the output is a list of :math:`D` arrays with the shape of ``x``
      (``to_list=True``), or a single array of shape :math:`(D, \dots)` (``to_list=False``).

    .. seealso::

        - :func:`pyzernike.zernike_evaluate` to evaluate a sum of polynomials in polar coordinates.
        - :func:`pyzernike.xy_zernike_fit` to compute the coefficients from sampled values.
        - The page :doc:`../../mathematical_description/elliptic_annulus` for the extension of the Zernike polynomials to the elliptic annulus.
        - The page :doc:`../../mathematical_description/coefficients` for the operations on the coefficients.

    Parameters
    ----------
    coefficients : Union[Sequence[float], numpy.ndarray]
        The coefficients :math:`c_j` in the OSA/ANSI ordering, as a 1D array of :math:`P = \frac{(N+1)(N+2)}{2}`
        finite real numbers.

    x : numpy.ndarray
        The x Cartesian coordinates with any shape ``(...)``.

    y : numpy.ndarray
        The y Cartesian coordinates with the same shape as ``x``.
        If ``x`` and ``y`` are floating point arrays of the same dtype (e.g. ``numpy.float32``), this
        dtype is kept. Otherwise, both are converted to ``numpy.float64``.

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
        Only used if ``use_recursive`` is False. If True, the powers and the cosine and sine terms are cached and
        shared between the polynomials. Default is True.

    use_recursive : bool, optional
        If True, the polynomials are computed with the recurrence relation (recommended for high orders, see
        :func:`pyzernike.xy_zernike_polynomial_up_to_order`). Default is False.

    normalize : bool, optional
        If True, the coefficients are those of the polynomials normalized by the factors of Noll
        :math:`N_{n}^{m} = \sqrt{(2 - \delta_{m0})(n + 1)}` (see :func:`pyzernike.zernike_normalization`).
        Default is False.

    to_list : bool, optional
        Only used when several derivatives are computed. If True, the output is a list of arrays. If False, the
        output is a single array of shape :math:`(D, \dots)`. Default is True.

    None is accepted for all the parameters of the ellipse and for the derivative orders, and means the default value.

    Returns
    -------
    Union[numpy.ndarray, List[numpy.ndarray]]
        - If ``x_derivative`` and ``y_derivative`` are integers: an array with the shape of ``x``.
        - Otherwise, if ``to_list`` is True: a list of :math:`D` arrays with the shape of ``x``.
        - Otherwise: an array of shape :math:`(D, \dots)`, where ``output[d]`` is the derivative
          ``(x_derivative[d], y_derivative[d])``.

        The dtype is the common floating point dtype of ``x`` and ``y`` (``numpy.float64`` by default).

    Raises
    ------
    TypeError
        If ``coefficients``, ``x`` or ``y`` can not be converted to a floating point array, if a parameter of the
        ellipse is not a real number, if ``x_derivative`` or ``y_derivative`` is not an integer or a sequence of
        integers, if ``default`` is not an integer or a floating point number, or if ``precompute``,
        ``use_recursive``, ``normalize`` or ``to_list`` is not a boolean.

    ValueError
        If ``coefficients`` is not a 1D array of finite values whose length is :math:`\frac{(N+1)(N+2)}{2}`, if
        ``x`` and ``y`` do not have the same shape, if a parameter of the ellipse is not finite, if ``Rx`` or ``Ry``
        is not strictly positive, if ``h`` is not in :math:`[0, 1)`, if the sequences do not have the same length,
        or if a derivative order is negative.

    Examples
    --------
    Evaluate a wavefront and its slopes on an elliptic pupil:

    .. code-block:: python

        import numpy
        from pyzernike import xy_zernike_evaluate

        coefficients = numpy.zeros(15)  # all the polynomials up to the order 4
        coefficients[[4, 5, 8]] = [0.3, -0.15, 0.1]  # Z_2^0, Z_2^2 and Z_3^1

        pupil = dict(Rx=1.0, Ry=0.7, alpha=numpy.deg2rad(30.0))
        X, Y = numpy.meshgrid(numpy.linspace(-1, 1, 200), numpy.linspace(-1, 1, 200))
        W = xy_zernike_evaluate(coefficients, X, Y, **pupil)
        dW_dx, dW_dy = xy_zernike_evaluate(coefficients, X, Y, x_derivative=[1, 0], y_derivative=[0, 1], **pupil)

    """
    coefficients, order = _as_osa_coefficients(coefficients)
    x, y = _as_common_floating_arrays(x=x, y=y)
    Rx, Ry, x0, y0, alpha, h = _as_ellipse_parameters(Rx, Ry, x0, y0, alpha, h)
    default = _as_default_value(default)
    precompute = _check_boolean(precompute, "precompute")
    use_recursive = _check_boolean(use_recursive, "use_recursive")
    normalize = _check_boolean(normalize, "normalize")
    to_list = _check_boolean(to_list, "to_list")
    derivatives, single = _broadcast_indices(
        integers={},
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

    output = numpy.full((len(derivatives["x_derivative"]),) + x.shape, default, dtype=x.dtype)
    output[:, valid_mask] = _core_xy_evaluate(
        rho=rho_eq[valid_mask],
        theta=theta_eq[valid_mask],
        coefficients=coefficients,
        order=order,
        x_derivative=derivatives["x_derivative"],
        y_derivative=derivatives["y_derivative"],
        Rx=Rx,
        Ry=Ry,
        alpha=alpha,
        h=h,
        use_recursive=use_recursive,
        normalize=normalize,
        precompute=precompute,
    )

    return _format_output(output, single, to_list)