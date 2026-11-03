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

import warnings
from numbers import Integral, Real
from typing import Optional, Tuple, Union

import numpy

from ..core.parsing import _as_common_floating_arrays, _as_ellipse_parameters, _as_fit_weights, _as_order, _check_boolean
from ..core.core_cartesian_to_elliptic_annulus import _core_cartesian_to_elliptic_annulus
from ..core.core_fit import _core_fit
from ..core.core_indexing import _core_polynomials_count


def xy_zernike_fit(
    x: numpy.ndarray,
    y: numpy.ndarray,
    values: numpy.ndarray,
    order: Integral,
    *,
    Rx: Optional[Real] = 1.0,
    Ry: Optional[Real] = 1.0,
    x0: Optional[Real] = 0.0,
    y0: Optional[Real] = 0.0,
    alpha: Optional[Real] = 0.0,
    h: Optional[Real] = 0.0,
    weights: Optional[numpy.ndarray] = None,
    use_recursive: bool = False,
    normalize: bool = False,
    return_residuals: bool = False,
) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
    r"""
    Fit a function sampled in Cartesian coordinates on an elliptic annulus with the Zernike polynomials up to a
    radial order :math:`N` by linear least squares.

    .. figure:: ../../../pyzernike/resources/elliptic_annulus_domain.png
        :width: 400px
        :align: center

        The parameters of the elliptic annulus domain.

    The function :math:`f` sampled at the points :math:`(x_k, y_k)` is approximated by:

    .. math::

        f(x, y) \approx \sum_{j=0}^{P-1} c_j Zxy_j(x, y)

    where :math:`Zxy_j(x, y) = Z_j(\rho_{eq}(x, y), \theta_{eq}(x, y))` is the Zernike polynomial of OSA/ANSI
    index :math:`j` on the elliptic annulus of parameters :math:`(R_x, R_y, x_0, y_0, \alpha, h)` (see
    :func:`pyzernike.xy_zernike_polynomial`) and :math:`P = \frac{(N+1)(N+2)}{2}`. The coefficients :math:`c_j`
    minimize the weighted sum of the squared residuals:

    .. math::

        \min_{c} \sum_{k} w_k \left( f_k - \sum_{j} c_j Zxy_j(x_k, y_k) \right)^2

    The points are converted to the equivalent polar coordinates :math:`(\rho_{eq}, \theta_{eq})` (see
    :func:`pyzernike.cartesian_to_elliptic_annulus`), and the problem is solved as in :func:`pyzernike.zernike_fit`.
    The parameters of the annulus are fixed: their estimation is a non-linear problem (see
    :func:`pyzernike.xy_zernike_least_squares`).

    The points outside the elliptic annulus, or where :math:`x`, :math:`y`, ``values`` or ``weights`` is not
    finite, or where the weight is zero, are ignored.

    .. note::

        For a full ellipse (:math:`h = 0`), the mapping onto the unit disk is affine, so the normalized
        polynomials stay orthonormal for the inner product averaged over the area of the ellipse: with
        ``normalize=True`` and a sampling dense and uniform in area, the variance of :math:`f` on the ellipse is
        :math:`\sum_{j > 0} c_j^2`. For :math:`h > 0`, the polynomials are orthonormal only for a non-uniform
        weight, and this relation is approximate (see the page :doc:`../../mathematical_description/fitting`).

    .. seealso::

        - :func:`pyzernike.zernike_fit` to fit a function sampled in polar coordinates on the unit disk.
        - :func:`pyzernike.xy_zernike_evaluate` to evaluate the fitted function and its derivatives from the
          coefficients.
        - :func:`pyzernike.xy_zernike_least_squares` to estimate also the parameters of the annulus.
        - :func:`pyzernike.zernike_index_to_order` to convert the OSA/ANSI indices to orders :math:`(n, m)`.
        - The page :doc:`../../mathematical_description/fitting` for the least squares fit and the orthogonality on the annulus.

    Parameters
    ----------
    x : numpy.ndarray
        The x Cartesian coordinates of the samples, with any shape ``(...)``.

    y : numpy.ndarray
        The y Cartesian coordinates of the samples, with the same shape as ``x``.

    values : numpy.ndarray
        The sampled values :math:`f_k`, with the same shape as ``x``.

    order : Integral
        The maximum radial order :math:`N \geq 0` of the fit.

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

    weights : Optional[numpy.ndarray], optional
        The weights :math:`w_k \geq 0` of the samples, with the same shape as ``x``. Default is None
        (all the weights are equal to 1).

    use_recursive : bool, optional
        If True, the polynomials are computed with the recurrence relation (recommended for high orders,
        see :func:`pyzernike.xy_zernike_polynomial_up_to_order`). Default is False.

    normalize : bool, optional
        If True, the fit uses the polynomials normalized by the factors of Noll
        :math:`N_{n}^{m} = \sqrt{(2 - \delta_{m0})(n + 1)}`, and the coefficients are those of the
        normalized basis. Default is False.

    return_residuals : bool, optional
        If True, the residuals :math:`f_k - \sum_{j} c_j Zxy_j(x_k, y_k)` are also returned. Default is False.

    None is accepted for all the parameters of the ellipse, and means the default value.

    Returns
    -------
    coefficients : numpy.ndarray
        The coefficients :math:`c_j` with shape :math:`(P,)`, in the OSA/ANSI ordering, as ``numpy.float64``.

    residuals : numpy.ndarray
        Only if ``return_residuals`` is True. The residuals with the shape of ``x``, as ``numpy.float64``.
        They are set to ``numpy.nan`` at the ignored points.

    Raises
    ------
    TypeError
        If ``x``, ``y``, ``values`` or ``weights`` can not be converted to a floating point array, if
        ``order`` is not an integer, if a parameter of the ellipse is not a real number, or if
        ``use_recursive``, ``normalize`` or ``return_residuals`` is not a boolean.

    ValueError
        If the arrays do not have the same shape, if ``order`` is negative, if a parameter of the ellipse is
        not finite, if ``Rx`` or ``Ry`` is not strictly positive, if ``h`` is not in :math:`[0, 1)`, if a
        weight is negative, or if the number of valid samples is lower than the number :math:`P` of polynomials.

    Warns
    -----
    RuntimeWarning
        If the design matrix is rank deficient (the samples do not determine all the coefficients, for
        example if they do not cover the annulus). The minimum norm solution is then returned.

    Examples
    --------
    Fit a wavefront measured on an off-axis (elliptic) pupil up to the order 4:

    .. code-block:: python

        import numpy
        from pyzernike import xy_zernike_fit, xy_zernike_polynomial

        pupil = dict(Rx=1.0, Ry=0.7, alpha=numpy.deg2rad(30.0))
        X, Y = numpy.meshgrid(numpy.linspace(-1, 1, 200), numpy.linspace(-1, 1, 200))

        W = 0.3 * xy_zernike_polynomial(X, Y, n=2, m=0, **pupil) - 0.1 * xy_zernike_polynomial(X, Y, n=3, m=1, **pupil)
        coefficients = xy_zernike_fit(X, Y, W, order=4, **pupil)  # NaN outside the pupil are ignored
        # coefficients[4] ~ 0.3 (Z_2^0) and coefficients[8] ~ -0.1 (Z_3^1)

    Evaluate the fitted wavefront and its slopes:

    .. code-block:: python

        from pyzernike import xy_zernike_evaluate

        fitted = xy_zernike_evaluate(coefficients, X, Y, **pupil)
        dW_dx, dW_dy = xy_zernike_evaluate(coefficients, X, Y, x_derivative=[1, 0], y_derivative=[0, 1], **pupil)

    """
    x, y, values = _as_common_floating_arrays(x=x, y=y, values=values)
    weights = _as_fit_weights(weights, shape=x.shape)
    order = _as_order(order)
    Rx, Ry, x0, y0, alpha, h = _as_ellipse_parameters(Rx, Ry, x0, y0, alpha, h)
    use_recursive = _check_boolean(use_recursive, "use_recursive")
    normalize = _check_boolean(normalize, "normalize")
    return_residuals = _check_boolean(return_residuals, "return_residuals")

    n_polynomials = _core_polynomials_count(order)

    x = x.astype(numpy.float64)
    y = y.astype(numpy.float64)
    rho_list, theta_list = _core_cartesian_to_elliptic_annulus(
        x=x,
        y=y,
        Rx=Rx,
        Ry=Ry,
        x0=x0,
        y0=y0,
        alpha=alpha,
        h=h,
        x_derivative=[0],
        y_derivative=[0],
    )
    rho_eq, theta_eq = rho_list[0], theta_list[0]

    valid_mask = (
        numpy.isfinite(rho_eq) & numpy.isfinite(theta_eq) & numpy.isfinite(values) & numpy.isfinite(weights)
        & (rho_eq >= 0) & (rho_eq <= 1) & (weights > 0)
    )
    n_valid = int(valid_mask.sum())
    if n_valid < n_polynomials:
        raise ValueError(
            f"The number of valid samples ({n_valid}) must be at least the number of polynomials ({n_polynomials}) "
            f"up to the order {order}."
        )

    coefficients, fitted, rank = _core_fit(
        rho=rho_eq[valid_mask],
        theta=theta_eq[valid_mask],
        values=values[valid_mask].astype(numpy.float64),
        weights=weights[valid_mask].astype(numpy.float64),
        order=order,
        use_recursive=use_recursive,
        normalize=normalize,
    )
    if rank < n_polynomials:
        warnings.warn(
            f"The design matrix is rank deficient (rank {rank} < {n_polynomials}): the samples do not determine "
            "all the coefficients, the minimum norm solution is returned.",
            RuntimeWarning,
            stacklevel=2,
        )

    if not return_residuals:
        return coefficients

    residuals = numpy.full(x.shape, numpy.nan, dtype=numpy.float64)
    residuals[valid_mask] = values[valid_mask].astype(numpy.float64) - fitted
    return coefficients, residuals