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
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import numpy

from ..core.parsing import (
    _as_choices,
    _as_common_floating_arrays,
    _as_ellipse_parameters,
    _as_fit_weights,
    _as_order,
    _as_osa_coefficients,
    _check_boolean,
)
from ..core.core_indexing import _core_polynomials_count
from ..core.core_least_squares import _core_xy_least_squares
from ..core.core_xy_polynomial import ELLIPSE_PARAMETERS


def xy_zernike_least_squares(
    x: numpy.ndarray,
    y: numpy.ndarray,
    values: numpy.ndarray,
    order: Integral,
    parameters: Union[str, Sequence[str]] = ("x0", "y0", "Rx", "Ry"),
    *,
    fit_coefficients: bool = False,
    coefficients: Optional[Union[Sequence[float], numpy.ndarray]] = None,
    Rx: Optional[Real] = 1.0,
    Ry: Optional[Real] = 1.0,
    x0: Optional[Real] = 0.0,
    y0: Optional[Real] = 0.0,
    alpha: Optional[Real] = 0.0,
    h: Optional[Real] = 0.0,
    weights: Optional[numpy.ndarray] = None,
    normalize: bool = False,
    use_recursive: bool = False,
    extrapolate: bool = True,
) -> Tuple[Callable[[numpy.ndarray], numpy.ndarray], Callable[[numpy.ndarray], numpy.ndarray]]:
    r"""
    Build the residual and the Jacobian functions of the least squares fit of a sampled function with Zernike
    polynomials on an elliptic annulus whose parameters are also unknown, to be given to any least squares solver.

    .. figure:: ../../../pyzernike/resources/elliptic_annulus_domain.png
        :width: 400px
        :align: center

        The parameters of the elliptic annulus domain.

    The function :math:`f` sampled at the points :math:`(x_k, y_k)` is modeled by:

    .. math::

        W(x, y; c, p) = \sum_{j=0}^{P-1} c_j Zxy_j(x, y; p)

    where :math:`Zxy_j(\cdot; p)` is the Zernike polynomial of OSA/ANSI index :math:`j` on the elliptic annulus of
    parameters :math:`p` (see :func:`pyzernike.xy_zernike_polynomial`) and :math:`P = \frac{(N+1)(N+2)}{2}`. The
    unknowns are gathered in the vector:

    .. math::

        q = (p_1, \dots, p_Q) \quad \text{(or } q = (c_0, \dots, c_{P-1}, p_1, \dots, p_Q) \text{ if ``fit_coefficients=True``)}

    where :math:`p_1, \dots, p_Q` are the free parameters of the annulus given by ``parameters``, in this order. The
    other parameters keep the values given by ``Rx``, ``Ry``, ``x0``, ``y0``, ``alpha`` and ``h``. The returned
    functions compute the weighted residuals and their Jacobian:

    .. math::

        r_k(q) = \sqrt{w_k} \left( W(x_k, y_k; q) - f_k \right)
        \quad \text{and} \quad
        J_{ki}(q) = \frac{\partial r_k}{\partial q_i}

    with :math:`\frac{\partial W}{\partial c_j} = Zxy_j` and
    :math:`\frac{\partial W}{\partial p_i} = \sum_j c_j \frac{\partial Zxy_j}{\partial p_i}` (exact, see
    :func:`pyzernike.xy_zernike_parameter_derivative`). Any solver of non-linear least squares problems can then
    minimize :math:`\frac{1}{2} \| r(q) \|^2`: ``scipy.optimize.least_squares(residual, q0, jac=jacobian)``, a
    Gauss-Newton or Levenberg-Marquardt algorithm, ``pysolve-gn``, ...

    **Samples.** The residuals are computed at the valid samples only, i.e. where ``x``, ``y``, ``values`` and
    ``weights`` are finite and the weight is strictly positive, in the order of the flattened arrays: the residual
    vector has the length :math:`K` of ``x[mask]`` (``mask`` the boolean array of the valid samples) for all :math:`q`.

    **Identifiability.** A shift, a scaling or a rotation of the pupil transforms a polynomial of degree
    :math:`N` into another polynomial of degree :math:`N`: for a full ellipse (:math:`h = 0`), a change of the
    parameters :math:`(x_0, y_0, R_x, R_y, \alpha)` is exactly compensated by a change of the coefficients (see
    :func:`pyzernike.zernike_coefficients_transform`). These parameters can therefore be estimated from the values of
    the function only if the coefficients are known (``fit_coefficients=False``, e.g. the known aberrations of a
    reference optics). Fitting them jointly with the coefficients gives a singular Jacobian for :math:`h = 0` (a
    ``RuntimeWarning`` is emitted), and a poorly conditioned problem for :math:`h > 0`. The position and the size of
    the pupil are then better measured from its boundary (e.g. on an intensity image). See the page
    :doc:`../../mathematical_description/fitting` for the details.

    **Points outside the annulus.** The set of the points inside the annulus changes with :math:`p`, which is not
    differentiable. With ``extrapolate=True`` (recommended), the polynomials are evaluated with their polynomial
    expression at all the samples, also slightly outside the current annulus, so that :math:`r(q)` is a smooth
    function of :math:`q` and all the samples always contribute. With ``extrapolate=False``, the residuals and the
    rows of the Jacobian of the samples outside the current annulus are set to zero: the cost can then decrease by
    shrinking the annulus, so this option should only be used with a good initial guess.

    .. note::

        The functions raise a ``ValueError`` if :math:`q` gives an invalid annulus (:math:`R_x \leq 0`,
        :math:`R_y \leq 0` or :math:`h` outside :math:`[0, 1)`). With ``scipy.optimize.least_squares``, use the
        ``bounds`` argument to keep the parameters in their domain. The last evaluation is cached: calling the
        Jacobian after the residuals at the same :math:`q` does not compute the polynomials again.

    .. seealso::

        - :func:`pyzernike.xy_zernike_fit` to fit the coefficients with a fixed annulus (linear problem).
        - :func:`pyzernike.xy_zernike_parameter_derivative` for the derivatives of the polynomials with respect to
          the parameters of the annulus.
        - :func:`pyzernike.zernike_coefficients_transform` for the change of pupil of a vector of coefficients.
        - The page :doc:`../../mathematical_description/fitting` for the non-linear least squares problem and its identifiability.

    Parameters
    ----------
    x : numpy.ndarray
        The x Cartesian coordinates of the samples, with any shape ``(...)``.

    y : numpy.ndarray
        The y Cartesian coordinates of the samples, with the same shape as ``x``.

    values : numpy.ndarray
        The sampled values :math:`f_k`, with the same shape as ``x``.

    order : Integral
        The maximum radial order :math:`N \geq 0` of the model.

    parameters : Union[str, Sequence[str]], optional
        The free parameter(s) of the annulus, among ``"x0"``, ``"y0"``, ``"Rx"``, ``"Ry"``, ``"alpha"`` and ``"h"``,
        without duplicates, in the order of the vector of unknowns. An empty sequence means that the annulus is
        fixed. Default is ``("x0", "y0", "Rx", "Ry")``.

    fit_coefficients : bool, optional
        If False, the coefficients are fixed to ``coefficients``. If True, the coefficients are unknowns, placed before
        the parameters of the annulus in :math:`q` (see the identifiability above). Default is False.

    coefficients : Optional[Union[Sequence[float], numpy.ndarray]], optional
        The fixed coefficients in the OSA/ANSI ordering, as a 1D array of :math:`P = \frac{(N+1)(N+2)}{2}` finite
        real numbers. Required if ``fit_coefficients`` is False (default), and must be None otherwise. Default is None.

    Rx : Optional[Real], optional
        The semi-axis :math:`R_x` of the ellipse along its x axis (fixed value, or ignored if free). Default is 1.0.

    Ry : Optional[Real], optional
        The semi-axis :math:`R_y` of the ellipse along its y axis (fixed value, or ignored if free). Default is 1.0.

    x0 : Optional[Real], optional
        The x coordinate :math:`x_0` of the center of the ellipse (fixed value, or ignored if free). Default is 0.0.

    y0 : Optional[Real], optional
        The y coordinate :math:`y_0` of the center of the ellipse (fixed value, or ignored if free). Default is 0.0.

    alpha : Optional[Real], optional
        The rotation angle :math:`\alpha` of the ellipse in radians (fixed value, or ignored if free). Default is 0.0.

    h : Optional[Real], optional
        The ratio :math:`h` of the inner to the outer semi-axes, in :math:`[0, 1)` (fixed value, or ignored if free).
        Default is 0.0 (full ellipse).

    weights : Optional[numpy.ndarray], optional
        The weights :math:`w_k \geq 0` of the samples, with the same shape as ``x``. Default is None (all the
        weights are equal to 1).

    normalize : bool, optional
        If True, the coefficients are those of the polynomials normalized by the factors of Noll
        :math:`N_{n}^{m} = \sqrt{(2 - \delta_{m0})(n + 1)}`. Default is False.

    use_recursive : bool, optional
        If True, the polynomials are computed with the recurrence relation (recommended for high orders).
        Default is False.

    extrapolate : bool, optional
        If True, the model is extrapolated outside the current annulus. If False, the samples outside the current
        annulus are ignored (zero residuals and zero rows of the Jacobian). Default is True.

    None is accepted for all the parameters of the ellipse, and means the default value.

    Returns
    -------
    residual : Callable[[numpy.ndarray], numpy.ndarray]
        The function :math:`q \mapsto r(q)`, returning a new array of shape :math:`(K,)`.

    jacobian : Callable[[numpy.ndarray], numpy.ndarray]
        The function :math:`q \mapsto J(q)`, returning a new array of shape :math:`(K, \dim q)` with
        :math:`\dim q = Q` (or :math:`P + Q` if ``fit_coefficients`` is True).

    Warns
    -----
    RuntimeWarning
        If ``fit_coefficients`` is True, :math:`h = 0` is fixed and a parameter among :math:`x_0, y_0, R_x, R_y,
        \alpha` is free: the problem is not identifiable (singular Jacobian).

    Raises
    ------
    TypeError
        If ``x``, ``y``, ``values``, ``weights`` or ``coefficients`` can not be converted to a floating point array,
        if ``order`` is not an integer, if ``parameters`` is not a string or a sequence of strings, if a parameter of
        the ellipse is not a real number, or if ``fit_coefficients``, ``normalize``, ``use_recursive`` or
        ``extrapolate`` is not a boolean.

    ValueError
        If the arrays do not have the same shape, if ``order`` is negative, if ``parameters`` contains an unknown
        name or a duplicate, if there is no unknown, if ``coefficients`` is given with ``fit_coefficients=True`` or
        missing with ``fit_coefficients=False``, if its length is not :math:`P`, if a parameter of the ellipse is not
        valid, if a weight is negative, or if there is no valid sample. The returned functions raise a ``ValueError``
        if :math:`q` does not have the shape :math:`(\dim q,)`, is not finite, or gives an invalid annulus.

    Examples
    --------
    Estimate the position, the size and the orientation of a pupil from a wavefront measured through a reference
    optics with known aberrations ``known_coefficients`` (OSA/ANSI, order 4), with ``scipy.optimize.least_squares``:

    .. code-block:: python

        import numpy
        import scipy.optimize
        from pyzernike import xy_zernike_least_squares

        parameters = ("x0", "y0", "Rx", "Ry", "alpha")
        residual, jacobian = xy_zernike_least_squares(X, Y, measured, order=4, parameters=parameters, coefficients=known_coefficients)

        q0 = numpy.array([0.0, 0.0, 1.0, 1.0, 0.0])  # x0, y0, Rx, Ry, alpha
        bounds = ([-numpy.inf, -numpy.inf, 1e-6, 1e-6, -numpy.inf], numpy.inf)
        result = scipy.optimize.least_squares(residual, q0, jac=jacobian, bounds=bounds)
        x0, y0, Rx, Ry, alpha = result.x

    Estimate only the center of the pupil with a hand-written Gauss-Newton algorithm:

    .. code-block:: python

        residual, jacobian = xy_zernike_least_squares(X, Y, measured, order=4, parameters=("x0", "y0"), coefficients=known_coefficients)
        q = numpy.array([0.0, 0.0])
        for _ in range(10):
            q -= numpy.linalg.lstsq(jacobian(q), residual(q), rcond=None)[0]

    """
    x, y, values = _as_common_floating_arrays(x=x, y=y, values=values)
    weights = _as_fit_weights(weights, shape=x.shape)
    order = _as_order(order)
    Rx, Ry, x0, y0, alpha, h = _as_ellipse_parameters(Rx, Ry, x0, y0, alpha, h)
    fit_coefficients = _check_boolean(fit_coefficients, "fit_coefficients")
    normalize = _check_boolean(normalize, "normalize")
    use_recursive = _check_boolean(use_recursive, "use_recursive")
    extrapolate = _check_boolean(extrapolate, "extrapolate")

    if isinstance(parameters, (tuple, list)) and len(parameters) == 0:
        parameters = []
    else:
        parameters = _as_choices(parameters, "parameters", ELLIPSE_PARAMETERS)
        parameters = [parameters] if isinstance(parameters, str) else list(parameters)
    if len(set(parameters)) != len(parameters):
        raise ValueError(f"parameters must not contain duplicates (got {parameters}).")
    if not fit_coefficients and not parameters:
        raise ValueError("There is no unknown: set fit_coefficients=True or give at least one parameter.")

    if fit_coefficients:
        if coefficients is not None:
            raise ValueError("coefficients must be None when fit_coefficients is True (they are unknowns).")
    else:
        if coefficients is None:
            raise ValueError("coefficients must be given when fit_coefficients is False.")
        coefficients, coefficients_order = _as_osa_coefficients(coefficients)
        if coefficients_order != order:
            raise ValueError(
                f"The length of coefficients must be {_core_polynomials_count(order)} for the order {order} "
                f"(got {coefficients.size})."
            )

    affine = [name for name in parameters if name != "h"]
    if fit_coefficients and affine and "h" not in parameters and h == 0.0:
        warnings.warn(
            f"The coefficients and the parameters {affine} of a full ellipse (h = 0) are not identifiable together: a "
            "change of these parameters is exactly compensated by a change of the coefficients (singular Jacobian). "
            "Fix the coefficients (fit_coefficients=False) or these parameters.",
            RuntimeWarning,
            stacklevel=2,
        )

    valid_mask = (
        numpy.isfinite(x) & numpy.isfinite(y) & numpy.isfinite(values) & numpy.isfinite(weights) & (weights > 0)
    )
    if not valid_mask.any():
        raise ValueError("There is no valid sample (finite x, y, values and weights, with a strictly positive weight).")

    def check_pupil(pupil: Dict[str, float]) -> None:
        _as_ellipse_parameters(pupil["Rx"], pupil["Ry"], pupil["x0"], pupil["y0"], pupil["alpha"], pupil["h"])

    return _core_xy_least_squares(
        x=x[valid_mask].astype(numpy.float64),
        y=y[valid_mask].astype(numpy.float64),
        values=values[valid_mask].astype(numpy.float64),
        sqrt_weights=numpy.sqrt(weights[valid_mask].astype(numpy.float64)),
        order=order,
        parameters=parameters,
        fit_coefficients=fit_coefficients,
        coefficients=coefficients,
        pupil=dict(Rx=Rx, Ry=Ry, x0=x0, y0=y0, alpha=alpha, h=h),
        normalize=normalize,
        use_recursive=use_recursive,
        extrapolate=extrapolate,
        check_pupil=check_pupil,
    )