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

from typing import Callable, Dict, List, Optional, Tuple

import numpy

from .core_cartesian_to_elliptic_annulus import _core_cartesian_to_elliptic_annulus
from .core_indexing import _core_index_to_order, _core_polynomials_count
from .core_polynomial import _core_polynomials
from .core_recursive import _core_recursive_up_to_order
from .core_xy_polynomial import _core_xy_parameter_derivatives


def _core_xy_model(
    x: numpy.ndarray,
    y: numpy.ndarray,
    coefficients: numpy.ndarray,
    order: int,
    pupil: Dict[str, float],
    parameters: List[str],
    normalize: bool,
    use_recursive: bool,
    extrapolate: bool,
    jacobian: bool,
) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray], numpy.ndarray]:
    r"""
    Evaluate the model :math:`W(x, y; c, p) = \sum_j c_j Zxy_j(x, y; p)` on an elliptic annulus and its derivatives
    with respect to the coefficients :math:`c` and to the parameters :math:`p` of the annulus, without any check.

    .. warning::

        This method is an internal function of ``pyzernike`` that is not designed to be used by the users directly.
        Please use :func:`pyzernike.xy_zernike_least_squares`.

    The derivatives are:

    .. math::

        \frac{\partial W}{\partial c_j} = Zxy_j(x, y; p)
        \quad \text{and} \quad
        \frac{\partial W}{\partial p_i} = \sum_j c_j \frac{\partial Zxy_j}{\partial p_i}(x, y; p)

    where :math:`\frac{\partial Zxy_j}{\partial p_i}` is computed by
    :func:`pyzernike.core.core_xy_polynomial._core_xy_parameter_derivatives` (only for the non-zero coefficients).

    If ``extrapolate`` is True, the polynomials are evaluated with their polynomial expression at all the points,
    also outside the annulus (:math:`\rho_{eq} > 1`, or :math:`\rho_{eq} < 0` in the central hole), so that the
    model and its derivatives are smooth functions of :math:`p`. Otherwise, the points outside the annulus are
    flagged in the returned mask.

    Parameters
    ----------
    x : numpy.ndarray (1D array)
        The x Cartesian coordinates of the samples, as ``numpy.float64``.

    y : numpy.ndarray (1D array)
        The y Cartesian coordinates of the samples, with the same shape as ``x``.

    coefficients : numpy.ndarray (1D array)
        The coefficients :math:`c_j` in the OSA/ANSI ordering, with shape :math:`(P,)`.

    order : int
        The maximum radial order :math:`N`, with :math:`P = \frac{(N+1)(N+2)}{2}`.

    pupil : Dict[str, float]
        The valid parameters ``"Rx"``, ``"Ry"``, ``"x0"``, ``"y0"``, ``"alpha"`` and ``"h"`` of the annulus.

    parameters : List[str]
        The parameters :math:`p_i` with respect to which the model is derived (a subset of ``ELLIPSE_PARAMETERS``).

    normalize : bool
        If True, the coefficients are those of the polynomials normalized by the factors of Noll.

    use_recursive : bool
        If True, the polynomials are computed with the recurrence relation, otherwise with the explicit sum.

    extrapolate : bool
        If True, the model is evaluated at all the points. If False, the mask of the points inside the annulus is
        returned (the values outside are still computed by extrapolation).

    jacobian : bool
        If True, the derivatives are computed.

    Returns
    -------
    numpy.ndarray
        The model :math:`W` with the shape of ``x``.

    Optional[numpy.ndarray]
        The derivatives :math:`\frac{\partial W}{\partial c_j} = Zxy_j` with shape :math:`(P, K)`, or None if
        ``jacobian`` is False.

    Optional[numpy.ndarray]
        The derivatives :math:`\frac{\partial W}{\partial p_i}` with shape :math:`(Q, K)` (:math:`Q` the length of
        ``parameters``), or None if ``jacobian`` is False.

    numpy.ndarray
        The boolean mask of the points used: all True if ``extrapolate`` is True, otherwise the points inside the
        annulus (:math:`0 \leq \rho_{eq} \leq 1`).

    """
    n_polynomials = _core_polynomials_count(order)

    rho_list, theta_list = _core_cartesian_to_elliptic_annulus(
        x=x,
        y=y,
        Rx=pupil["Rx"],
        Ry=pupil["Ry"],
        x0=pupil["x0"],
        y0=pupil["y0"],
        alpha=pupil["alpha"],
        h=pupil["h"],
        x_derivative=[0],
        y_derivative=[0],
    )
    rho, theta = rho_list[0], theta_list[0]
    inside = numpy.ones(rho.shape, dtype=bool) if extrapolate else (rho >= 0) & (rho <= 1)

    if use_recursive:
        polynomials = _core_recursive_up_to_order(
            rho=rho,
            theta=theta,
            order=order,
            rho_derivative=[0],
            theta_derivative=[0],
            normalize=normalize,
        )[0]
    else:
        n, m = _core_index_to_order(list(range(n_polynomials)), indexing="OSA")
        polynomials = _core_polynomials(
            rho=rho,
            theta=theta,
            n=n,
            m=m,
            rho_derivative=[0] * n_polynomials,
            theta_derivative=[0] * n_polynomials,
            flag_radial=False,
            precompute=True,
            normalize=normalize,
        )
    model = coefficients @ polynomials

    if not jacobian:
        return model, None, None, inside

    parameter_derivatives = numpy.zeros((len(parameters), rho.size), dtype=numpy.float64)
    active = numpy.flatnonzero(coefficients)
    if parameters and active.size > 0:
        n_active, m_active = _core_index_to_order(active.tolist(), indexing="OSA")
        derivatives = _core_xy_parameter_derivatives(
            rho=rho,
            theta=theta,
            n=n_active * len(parameters),
            m=m_active * len(parameters),
            parameter=[name for name in parameters for _ in range(active.size)],
            Rx=pupil["Rx"],
            Ry=pupil["Ry"],
            alpha=pupil["alpha"],
            h=pupil["h"],
            precompute=True,
            normalize=normalize,
        ).reshape((len(parameters), active.size, rho.size))
        parameter_derivatives = numpy.tensordot(coefficients[active], derivatives, axes=(0, 1))

    return model, polynomials, parameter_derivatives, inside


def _core_xy_least_squares(
    x: numpy.ndarray,
    y: numpy.ndarray,
    values: numpy.ndarray,
    sqrt_weights: numpy.ndarray,
    order: int,
    parameters: List[str],
    fit_coefficients: bool,
    coefficients: Optional[numpy.ndarray],
    pupil: Dict[str, float],
    normalize: bool,
    use_recursive: bool,
    extrapolate: bool,
    check_pupil: Callable[[Dict[str, float]], None],
) -> Tuple[Callable[[numpy.ndarray], numpy.ndarray], Callable[[numpy.ndarray], numpy.ndarray]]:
    r"""
    Build the residual and Jacobian functions of the weighted least squares problem on an elliptic annulus, without
    any check of the inputs.

    .. warning::

        This method is an internal function of ``pyzernike`` that is not designed to be used by the users directly.
        Please use :func:`pyzernike.xy_zernike_least_squares`.

    The vector of unknowns is :math:`q = (c_0, \dots, c_{P-1}, p_1, \dots, p_Q)` if ``fit_coefficients`` is True,
    and :math:`q = (p_1, \dots, p_Q)` otherwise (the coefficients are then fixed to ``coefficients``). The residuals
    are :math:`r_k(q) = \sqrt{w_k} \left( W(x_k, y_k; q) - f_k \right)` and the Jacobian is
    :math:`J_{ki} = \frac{\partial r_k}{\partial q_i}`. The last evaluation is cached, so that a call to the Jacobian
    after a call to the residuals at the same :math:`q` does not compute the polynomials again.

    Parameters
    ----------
    x : numpy.ndarray (1D array)
        The x Cartesian coordinates of the valid samples, as ``numpy.float64``.

    y : numpy.ndarray (1D array)
        The y Cartesian coordinates of the valid samples, with the same shape as ``x``.

    values : numpy.ndarray (1D array)
        The finite sampled values :math:`f_k`, with the same shape as ``x``.

    sqrt_weights : numpy.ndarray (1D array)
        The square roots :math:`\sqrt{w_k} > 0` of the weights, with the same shape as ``x``.

    order : int
        The maximum radial order :math:`N`.

    parameters : List[str]
        The free parameters of the annulus, in the order of the vector of unknowns.

    fit_coefficients : bool
        If True, the coefficients are unknowns, placed before the parameters of the annulus.

    coefficients : Optional[numpy.ndarray] (1D array)
        The fixed coefficients with shape :math:`(P,)`, used only if ``fit_coefficients`` is False.

    pupil : Dict[str, float]
        The values of all the parameters of the annulus. The free parameters are replaced by the values of the
        vector of unknowns at each evaluation.

    normalize : bool
        If True, the coefficients are those of the polynomials normalized by the factors of Noll.

    use_recursive : bool
        If True, the polynomials are computed with the recurrence relation, otherwise with the explicit sum.

    extrapolate : bool
        If True, the model is extrapolated outside the annulus. If False, the residuals and the rows of the Jacobian
        of the samples outside the current annulus are set to zero.

    check_pupil : Callable[[Dict[str, float]], None]
        A function that raises a ``ValueError`` if the parameters of the annulus are not valid.

    Returns
    -------
    Callable[[numpy.ndarray], numpy.ndarray]
        The residual function :math:`q \mapsto r(q)`, with an output of shape :math:`(K,)`.

    Callable[[numpy.ndarray], numpy.ndarray]
        The Jacobian function :math:`q \mapsto J(q)`, with an output of shape :math:`(K, \dim q)`.

    """
    n_polynomials = _core_polynomials_count(order)
    n_unknowns = (n_polynomials if fit_coefficients else 0) + len(parameters)
    cache: Dict[str, object] = {"key": None, "residual": None, "jacobian": None}

    def _unpack(q: numpy.ndarray) -> Tuple[numpy.ndarray, Dict[str, float]]:
        q = numpy.asarray(q, dtype=numpy.float64)
        if q.shape != (n_unknowns,):
            raise ValueError(f"The vector of unknowns must have the shape ({n_unknowns},) (got {q.shape}).")
        if not numpy.all(numpy.isfinite(q)):
            raise ValueError("The vector of unknowns must contain finite values.")
        current = dict(pupil)
        offset = n_polynomials if fit_coefficients else 0
        for index, name in enumerate(parameters):
            current[name] = float(q[offset + index])
        check_pupil(current)
        return (q[:n_polynomials] if fit_coefficients else coefficients), current

    def _evaluate(q: numpy.ndarray, jacobian: bool) -> None:
        key = numpy.asarray(q, dtype=numpy.float64).tobytes()
        if cache["key"] == key and (cache["jacobian"] is not None or not jacobian):
            return
        current_coefficients, current_pupil = _unpack(q)
        model, polynomials, parameter_derivatives, inside = _core_xy_model(
            x=x,
            y=y,
            coefficients=current_coefficients,
            order=order,
            pupil=current_pupil,
            parameters=parameters,
            normalize=normalize,
            use_recursive=use_recursive,
            extrapolate=extrapolate,
            jacobian=jacobian,
        )
        residual = numpy.where(inside, sqrt_weights * (model - values), 0.0)
        matrix = None
        if jacobian:
            blocks = ([polynomials] if fit_coefficients else []) + [parameter_derivatives]
            matrix = (numpy.concatenate(blocks, axis=0) * numpy.where(inside, sqrt_weights, 0.0)).T
        cache.update(key=key, residual=residual, jacobian=matrix)

    def residual_function(q: numpy.ndarray) -> numpy.ndarray:
        _evaluate(q, jacobian=False)
        return cache["residual"].copy()

    def jacobian_function(q: numpy.ndarray) -> numpy.ndarray:
        _evaluate(q, jacobian=True)
        return cache["jacobian"].copy()

    return residual_function, jacobian_function