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

r"""
Computation of all the Zernike polynomials up to an order on an elliptic annulus, and of their derivatives with
respect to :math:`x` and :math:`y`, with the recurrence relation of Shakibaei and Paramesran
(see :func:`pyzernike.core._core_recursive_up_to_order`) instead of the explicit sums, which lose their accuracy
beyond :math:`n \approx 20`.

The exact integer coefficients of the derivatives are built symbolically (as in :mod:`pyzernike.core.core_xy_terms`), but
they are applied to values computed by the recurrence relation instead of powers of :math:`\rho`:

- For :math:`h = 0`, the derivatives are linear combinations of Zernike polynomials of lower orders (Noll), with
  exact rational coefficients (see :func:`_zernike_expansion`).
- For :math:`h > 0`, the derivatives are sums of terms :math:`s^{a} \frac{d^a R_{n}^{|m|}}{d\rho^a}(\rho_{eq}) \, r^{-q} \cos(k\theta)`
  (or :math:`\sin`) with exact integer coefficients (see :func:`_annulus_derivative_terms`).
"""

from collections import defaultdict
from fractions import Fraction
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy

from .core_normalization import _core_normalization
from .core_indexing import _core_index_to_order, _core_order_to_index, _core_polynomials_count
from .core_xy_polynomial import _uv_weights
from .core_polynomial import _radial_coefficients
from .core_xy_terms import Terms, _XY_TERMS_UV_RULES, _xy_terms_accumulate, _xy_terms_freeze, _xy_terms_merge_full_ellipse, _xy_terms_uv_derivative
from .core_recursive import _core_recursive_up_to_order


@lru_cache(maxsize=None)
def _zernike_expansion(n: int, m: int, u_derivative: int, v_derivative: int) -> Tuple[Tuple[int, Fraction], ...]:
    r"""
    Expand :math:`2^{d} \partial_u^{d_u} \partial_v^{d_v} Z_{n}^{m}` (:math:`d = d_u + d_v`, full ellipse :math:`h = 0`)
    in the basis of the Zernike polynomials, with exact rational coefficients (Noll).

    The derivative is a sum of terms :math:`c \, r^{e} \cos(k\theta)` (or :math:`\sin`) with :math:`e \geq k` and
    :math:`e - k` even (see :func:`pyzernike.core.core_xy_terms._xy_terms_uv_derivative` and
    :func:`pyzernike.core.core_xy_terms._xy_terms_merge_full_ellipse`). Each radial polynomial :math:`\sum_e c_e r^e` is projected on the
    radial polynomials :math:`R_{e}^{k}`, by decreasing :math:`e`: the leading term :math:`c_e r^e` is
    :math:`\frac{c_e}{\gamma_e} R_{e}^{k}` minus lower powers, where :math:`\gamma_e` is the leading coefficient of :math:`R_{e}^{k}`.
    The coefficients are cached.

    Parameters
    ----------
    n : int
        The radial order.

    m : int
        The azimuthal frequency.

    u_derivative : int
        The order :math:`d_u` of the derivative with respect to :math:`u`.

    v_derivative : int
        The order :math:`d_v` of the derivative with respect to :math:`v`.

    Returns
    -------
    Tuple[Tuple[int, Fraction], ...]
        The pairs ``(j, c)`` of the OSA/ANSI indices and of the non-zero coefficients.

    Raises
    ------
    RuntimeError
        If a term can not be projected (should never happen).

    """
    expansion: Dict[int, Fraction] = defaultdict(Fraction)
    for kind, k, radial in _xy_terms_uv_derivative(n, m, u_derivative, v_derivative):
        polynomial = {e: Fraction(c) for (e, _), c in _xy_terms_merge_full_ellipse(radial)}
        while polynomial:
            e = max(polynomial)
            c = polynomial.pop(e)
            if c == 0:
                continue
            if e < k or (e - k) % 2 != 0:
                raise RuntimeError("Internal error: a term can not be projected on the Zernike polynomials.")
            coefficients = _radial_coefficients(e, k, 0)
            a = c / coefficients[0]
            for i, coefficient in enumerate(coefficients[1:], 1):
                polynomial[e - 2 * i] = polynomial.get(e - 2 * i, 0) - a * coefficient
            expansion[_core_order_to_index([e], [k if kind == "cos" else -k])[0]] += a
    return tuple(sorted((j, c) for j, c in expansion.items() if c != 0))


def _noll_matrix(order: int, x_derivative: int, y_derivative: int, Rx: float, Ry: float, alpha: float) -> numpy.ndarray:
    r"""
    Build the matrix :math:`C` such that :math:`\partial_x^{d_x} \partial_y^{d_y} Z_j = \sum_k C_{jk} Z_k` for all the
    polynomials up to the order :math:`N` (full ellipse :math:`h = 0`, OSA/ANSI ordering).

    The operator :math:`\partial_x^{d_x} \partial_y^{d_y}` is expanded in the operators :math:`\partial_u^{d_u} \partial_v^{d_v}`
    (see :func:`pyzernike.core.core_xy_polynomial._uv_weights`), and each derivative is expanded in the basis of the
    Zernike polynomials (see :func:`_zernike_expansion`).

    Parameters
    ----------
    order : int
        The maximum radial order :math:`N`.

    x_derivative : int
        The order :math:`d_x` of the derivative with respect to :math:`x`.

    y_derivative : int
        The order :math:`d_y` of the derivative with respect to :math:`y`.

    Rx : float
        The semi-axis :math:`R_x` of the ellipse along its x axis.

    Ry : float
        The semi-axis :math:`R_y` of the ellipse along its y axis.

    alpha : float
        The rotation angle :math:`\alpha` of the ellipse in radians.

    Returns
    -------
    numpy.ndarray
        The matrix :math:`C` of shape :math:`(P, P)` (``numpy.float64``), with :math:`P = \frac{(N+1)(N+2)}{2}`.

    """
    n_polynomials = _core_polynomials_count(order)
    n_list, m_list = _core_index_to_order(list(range(n_polynomials)))
    total_order = x_derivative + y_derivative
    matrix = numpy.zeros((n_polynomials, n_polynomials))
    for u_order, weight in _uv_weights(x_derivative, y_derivative, Rx, Ry, alpha).items():
        scale = weight * 0.5 ** total_order
        for row, (n, m) in enumerate(zip(n_list, m_list)):
            for column, coefficient in _zernike_expansion(n, m, u_order, total_order - u_order):
                matrix[row, column] += scale * float(coefficient)
    return matrix


def _annulus_operator(radial: Tuple[Tuple[Tuple[int, int], int], ...], k: int) -> Dict[Tuple[int, int], int]:
    r"""
    Apply :math:`A \mapsto A' + \frac{k}{r} A` (:math:`A' = \frac{dA}{dr}`) to a radial part written in the basis
    :math:`s^{a} R^{(a)}(\rho_{eq}) \, r^{-q}`, where :math:`R^{(a)} = \frac{d^a R_{n}^{|m|}}{d\rho^a}`,
    :math:`\rho_{eq} = \frac{r - h}{1 - h}` and :math:`s = \frac{1}{1 - h} = \frac{d\rho_{eq}}{dr}`:

    .. math::

        \frac{d}{dr} \left( s^{a} R^{(a)} r^{-q} \right) = s^{a+1} R^{(a+1)} r^{-q} - q \, s^{a} R^{(a)} r^{-(q+1)}

    so that :math:`(a, q) \mapsto (a + 1, q)` with the coefficient 1, and :math:`(a, q) \mapsto (a, q + 1)` with the
    coefficient :math:`k - q`. With :math:`-k`, the operator is :math:`A' - \frac{k}{r} A`.

    Parameters
    ----------
    radial : Tuple[Tuple[Tuple[int, int], int], ...]
        The monomials ``((a, q), c)`` of the radial part :math:`A`.

    k : int
        The (signed) factor of :math:`\frac{A}{r}`.

    Returns
    -------
    Dict[Tuple[int, int], int]
        The monomials of :math:`A' + \frac{k}{r} A`.

    """
    output: Dict[Tuple[int, int], int] = defaultdict(int)
    for (a, q), c in radial:
        output[(a + 1, q)] += c
        output[(a, q + 1)] += (k - q) * c
    return output


def _derive_annulus(terms: Terms, variable: str) -> Terms:
    r"""
    Apply :math:`2 \partial_u` or :math:`2 \partial_v` to terms written in the basis
    :math:`s^{a} R^{(a)}(\rho_{eq}) \, r^{-q} \cos(k\theta)` (or :math:`\sin`), with the rules of
    :func:`pyzernike.core.core_xy_terms._xy_terms_derive_uv` and the radial operator :func:`_annulus_operator`.

    Parameters
    ----------
    terms : Terms
        The terms to derive, with radial monomials ``((a, q), c)``.

    variable : str
        ``"u"`` or ``"v"``.

    Returns
    -------
    Terms
        The terms of the derivative, multiplied by 2.

    """
    accumulator = {}
    for kind, k, radial in terms:
        operators = {"P": _annulus_operator(radial, k), "M": _annulus_operator(radial, -k)}
        for new_kind, shift, sign, operator in _XY_TERMS_UV_RULES[(kind, variable)]:
            _xy_terms_accumulate(accumulator, new_kind, k + shift, sign, operators[operator])
    return _xy_terms_freeze(accumulator)


@lru_cache(maxsize=None)
def _annulus_derivative_terms(m: int, u_derivative: int, v_derivative: int) -> Terms:
    r"""
    Terms of :math:`2^{d} \partial_u^{d_u} \partial_v^{d_v} \left[ R_{n}^{|m|}(\rho_{eq}) \cos(m\theta) \right]` (or :math:`\sin`
    if :math:`m < 0`) in the basis :math:`s^{a} R^{(a)}(\rho_{eq}) \, r^{-q} \cos(k\theta)`, where
    :math:`R^{(a)} = \frac{d^a R_{n}^{|m|}}{d\rho^a}`. The terms do not depend on :math:`n` nor on the parameters of the
    annulus, and are cached.

    Parameters
    ----------
    m : int
        The azimuthal frequency.

    u_derivative : int
        The order :math:`d_u` of the derivative with respect to :math:`u`.

    v_derivative : int
        The order :math:`d_v` of the derivative with respect to :math:`v`.

    Returns
    -------
    Terms
        The terms ``(kind, k, ((a, q), c), ...)``.

    """
    if u_derivative > 0:
        return _derive_annulus(_annulus_derivative_terms(m, u_derivative - 1, v_derivative), "u")
    if v_derivative > 0:
        return _derive_annulus(_annulus_derivative_terms(m, 0, v_derivative - 1), "v")
    return (("cos" if m >= 0 else "sin", abs(m), (((0, 0), 1),)),)


def _normalize_output(output: numpy.ndarray, order: int, normalize: bool) -> numpy.ndarray:
    r"""
    Multiply in place the polynomials of all the orders up to :math:`N` by their normalization factors of Noll.

    Parameters
    ----------
    output : numpy.ndarray
        The polynomials with shape ``(D, P, ...)``, in the OSA/ANSI ordering. Modified in place.

    order : int
        The maximum radial order :math:`N`, with :math:`P = \frac{(N+1)(N+2)}{2}`.

    normalize : bool
        If False, ``output`` is returned unchanged.

    Returns
    -------
    numpy.ndarray
        The array ``output``.

    """
    if normalize:
        n_polynomials = _core_polynomials_count(order)
        n_list, m_list = _core_index_to_order(list(range(n_polynomials)), indexing="OSA")
        factors = numpy.asarray(_core_normalization(n_list, m_list), dtype=output.dtype)
        output *= factors.reshape((1, n_polynomials) + (1,) * (output.ndim - 2))
    return output


def _core_xy_recursive_up_to_order(
    rho: numpy.ndarray,
    theta: numpy.ndarray,
    order: int,
    x_derivative: List[int],
    y_derivative: List[int],
    Rx: float,
    Ry: float,
    alpha: float,
    h: float,
    normalize: bool,
) -> numpy.ndarray:
    r"""
    Compute all the Zernike polynomials up to the order :math:`N` on an elliptic annulus, and their derivatives with
    respect to :math:`x` and :math:`y`, with the recurrence relation of Shakibaei and Paramesran.

    - For :math:`h = 0`, the polynomials :math:`Z_k` are computed with
      :func:`pyzernike.core._core_recursive_up_to_order`, and the derivatives are obtained as
      :math:`\partial_x^{d_x} \partial_y^{d_y} Z_j = \sum_k C_{jk} Z_k` (see :func:`_noll_matrix`). There is no
      singularity at the center of the ellipse.
    - For :math:`h > 0`, the radial derivatives :math:`R^{(a)} = \frac{d^a R_{n}^{|m|}}{d\rho^a}(\rho_{eq})` are computed with
      the recurrence relation, and the derivatives are obtained as sums of terms
      :math:`c \, s^{a} R^{(a)} r^{-q} \cos(k\theta)` (or :math:`\sin`), with :math:`s = \frac{1}{1-h}` and
      :math:`r = h + (1 - h) \rho_{eq} \geq h > 0` (see :func:`_annulus_derivative_terms`).

    The coefficients are exact (integers or fractions) and do not depend on the points. The values combined by these
    coefficients are bounded and accurate at any order, so that the accuracy only degrades polynomially with the order
    (and not exponentially as with the explicit sums).

    Parameters
    ----------
    rho : numpy.ndarray (N-D array)
        The equivalent radial coordinate :math:`\rho_{eq}` of the points inside the elliptic annulus
        (:math:`0 \leq \rho_{eq} \leq 1`), with shape ``(...,)`` and floating point dtype.

    theta : numpy.ndarray (N-D array)
        The equivalent angular coordinate :math:`\theta_{eq}` with same shape and dtype as ``rho``.

    order : int
        The maximum radial order :math:`N \geq 0`.

    x_derivative : List[int]
        The non-negative orders of the derivatives with respect to :math:`x`.

    y_derivative : List[int]
        The non-negative orders of the derivatives with respect to :math:`y`. Must have the same length as ``x_derivative``.

    Rx : float
        The semi-axis :math:`R_x` of the ellipse along its x axis. Must be strictly positive.

    Ry : float
        The semi-axis :math:`R_y` of the ellipse along its y axis. Must be strictly positive.

    alpha : float
        The rotation angle :math:`\alpha` of the ellipse in radians.

    h : float
        The ratio :math:`h` of the inner to the outer semi-axes, in :math:`[0, 1)`.

    normalize : bool
        If True, the polynomials (and their derivatives) are multiplied by the normalization factors of Noll
        :math:`N_{n}^{m} = \sqrt{(2 - \delta_{m0})(n + 1)}` (see :func:`pyzernike.core.core_normalization._core_normalization`).

    Returns
    -------
    numpy.ndarray
        An array of shape ``(D, P, ...)`` and dtype of ``rho``, where :math:`D` is the length of ``x_derivative``
        and :math:`P = \frac{(N+1)(N+2)}{2}`. ``output[d, j]`` is the polynomial of OSA/ANSI index ``j`` for the pair
        ``(x_derivative[d], y_derivative[d])``.

    """
    dtype = rho.dtype
    n_polynomials = _core_polynomials_count(order)
    output = numpy.empty((len(x_derivative), n_polynomials) + rho.shape, dtype=dtype)
    if len(x_derivative) == 0:
        return output

    # ------------------------------------------------------------------ full ellipse: Noll matrices
    if h == 0.0:
        polynomials = _core_recursive_up_to_order(
            rho=rho,
            theta=theta,
            order=order,
            rho_derivative=[0],
            theta_derivative=[0],
            normalize=False,
        )[0].reshape((n_polynomials, -1))
        for index, (dx, dy) in enumerate(zip(x_derivative, y_derivative)):
            if dx == 0 and dy == 0:
                output[index] = polynomials.reshape((n_polynomials,) + rho.shape)
                continue
            matrix = _noll_matrix(order=order, x_derivative=dx, y_derivative=dy, Rx=Rx, Ry=Ry, alpha=alpha).astype(dtype)
            output[index] = (matrix @ polynomials).reshape((n_polynomials,) + rho.shape)
        return _normalize_output(output, order, normalize)

    # ------------------------------------------------------------------ elliptic annulus: radial derivatives
    max_order = max(dx + dy for dx, dy in zip(x_derivative, y_derivative))
    # With theta = 0, the output of the recurrence for (n, |m|) is the radial polynomial (cos(0) = 1)
    radial = _core_recursive_up_to_order(
        rho=rho,
        theta=numpy.zeros_like(rho),
        order=order,
        rho_derivative=list(range(max_order + 1)),
        theta_derivative=[0] * (max_order + 1),
        normalize=False,
    )

    # OSA/ANSI orders of the polynomials, and indices of their radial polynomials R_n^|m| in the output of the recurrence
    n_list, m_list = _core_index_to_order(list(range(n_polynomials)))
    radial_indices = _core_order_to_index(n_list, [abs(m) for m in m_list])

    s = 1.0 / (1.0 - h)
    inverse_r = 1.0 / (h + (1.0 - h) * rho)
    inverse_r_powers: Dict[int, numpy.ndarray] = {0: None, 1: inverse_r}
    angular: Dict[Tuple[str, int], numpy.ndarray] = {}

    def _inverse_r_power(q: int) -> numpy.ndarray:
        if q not in inverse_r_powers:
            inverse_r_powers[q] = _inverse_r_power(q - 1) * inverse_r
        return inverse_r_powers[q]

    def _angular(kind: str, k: int) -> numpy.ndarray:
        if (kind, k) not in angular:
            angular[(kind, k)] = numpy.cos(k * theta) if kind == "cos" else numpy.sin(k * theta)
        return angular[(kind, k)]

    for index, (dx, dy) in enumerate(zip(x_derivative, y_derivative)):
        total_order = dx + dy
        weights = _uv_weights(dx, dy, Rx, Ry, alpha)

        # The combined coefficients depend only on m (and not on n): computed once per m
        combined_by_m: Dict[int, Dict[Tuple[str, int, int, int], float]] = {}

        for j, (n, m) in enumerate(zip(n_list, m_list)):
            if m not in combined_by_m:
                combined: Dict[Tuple[str, int, int, int], float] = defaultdict(float)
                for u_order, weight in weights.items():
                    for kind, k, terms in _annulus_derivative_terms(m, u_order, total_order - u_order):
                        for (a, q), c in terms:
                            combined[(kind, k, a, q)] += weight * 0.5 ** total_order * s ** a * c
                combined_by_m[m] = {key: c for key, c in combined.items() if c != 0.0}

            value = numpy.zeros(rho.shape, dtype=dtype)
            radial_index = radial_indices[j]
            for (kind, k, a, q), c in combined_by_m[m].items():
                if a > n:
                    continue
                term = c * radial[a, radial_index]
                if q > 0:
                    term = term * _inverse_r_power(q)
                if not (kind == "cos" and k == 0):
                    term = term * _angular(kind, k)
                value += term
            output[index, j] = value

    return _normalize_output(output, order, normalize)