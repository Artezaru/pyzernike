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

import math
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy

from .core_normalization import _core_normalization
from .core_polynomial import _core_polynomials
from .core_xy_terms import XYFloatTerms, _XYTermsCache, _xy_terms_evaluate, _xy_terms_merge_full_ellipse, _xy_terms_uv_derivative

# The parameters of the elliptic annulus with respect to which the polynomials can be derived
ELLIPSE_PARAMETERS: Tuple[str, ...] = ("x0", "y0", "Rx", "Ry", "alpha", "h")


def _uv_weights(
    x_derivative: int,
    y_derivative: int,
    Rx: float,
    Ry: float,
    alpha: float,
) -> Dict[int, float]:
    r"""
    Expand :math:`\partial_x^{d_x} \partial_y^{d_y}` in the operators :math:`\partial_u^{d_u} \partial_v^{d - d_u}`
    (:math:`d = d_x + d_y`). The map :math:`(x, y) \mapsto (u, v)` is affine, so:

    .. math::

        \partial_x = \frac{\cos\alpha}{R_x} \partial_u - \frac{\sin\alpha}{R_y} \partial_v
        \quad \text{and} \quad
        \partial_y = \frac{\sin\alpha}{R_x} \partial_u + \frac{\cos\alpha}{R_y} \partial_v

    and the powers are expanded with the binomial formula.

    Parameters
    ----------
    x_derivative : int
        The order :math:`d_x \geq 0` of the derivative with respect to :math:`x`.

    y_derivative : int
        The order :math:`d_y \geq 0` of the derivative with respect to :math:`y`.

    Rx : float
        The semi-axis :math:`R_x` of the ellipse along its x axis.

    Ry : float
        The semi-axis :math:`R_y` of the ellipse along its y axis.

    alpha : float
        The rotation angle :math:`\alpha` of the ellipse in radians.

    Returns
    -------
    Dict[int, float]
        The non-zero weights of :math:`\partial_u^{d_u} \partial_v^{d - d_u}`, by :math:`d_u`.

    """
    du_dx, dv_dx = math.cos(alpha) / Rx, - math.sin(alpha) / Ry
    du_dy, dv_dy = math.sin(alpha) / Rx, math.cos(alpha) / Ry

    weights: Dict[int, float] = defaultdict(float)
    for i in range(x_derivative + 1):
        for j in range(y_derivative + 1):
            weight = (math.comb(x_derivative, i) * du_dx ** i * dv_dx ** (x_derivative - i)
                      * math.comb(y_derivative, j) * du_dy ** j * dv_dy ** (y_derivative - j))
            if weight != 0.0:
                weights[i + j] += weight
    return weights


def _core_xy_polynomials(
    rho: numpy.ndarray,
    theta: numpy.ndarray,
    n: List[int],
    m: List[int],
    x_derivative: List[int],
    y_derivative: List[int],
    Rx: float,
    Ry: float,
    alpha: float,
    h: float,
    precompute: bool,
    normalize: bool = False,
) -> numpy.ndarray:
    r"""
    Compute the derivatives with respect to :math:`x` and :math:`y` of the Zernike polynomials on an
    elliptic annulus, without singularity at the center of the ellipse.

    With :math:`u = X / R_x` and :math:`v = Y / R_y` (:math:`(X, Y)` the coordinates in the frame of the
    ellipse), :math:`r = \sqrt{u^2 + v^2}` and :math:`\theta = \operatorname{atan2}(v, u)`:

    1. :math:`\partial_x^{d_x} \partial_y^{d_y}` is expanded in the operators :math:`\partial_u^{d_u} \partial_v^{d_v}`
       (see :func:`_uv_weights`).

    2. The derivatives :math:`\partial_u^{d_u} \partial_v^{d_v} Z_{n}^{m}` are built symbolically with exact
       integer coefficients (see :func:`pyzernike.core.core_xy_terms._xy_terms_uv_derivative`), as sums of terms
       :math:`c \, s^{d-q} \rho_{eq}^{p} r^{-q} \cos(k\theta)` (or :math:`\sin`), with :math:`s = \frac{1}{1-h}`.

    3. The terms are combined and evaluated (see :func:`pyzernike.core.core_xy_terms._xy_terms_evaluate`).

    For :math:`h = 0`, :math:`\rho_{eq} = r` and the negative powers of :math:`r` cancel exactly
    (see :func:`pyzernike.core.core_xy_terms._xy_terms_merge_full_ellipse`): the result is a polynomial and is exact at
    :math:`r = 0`. For :math:`h > 0`, :math:`r \geq h > 0` on the elliptic annulus.

    Parameters
    ----------
    rho : numpy.ndarray (N-D array)
        The equivalent radial coordinate :math:`\rho_{eq}` of the points inside the elliptic annulus
        (:math:`0 \leq \rho_{eq} \leq 1`), with shape ``(...,)`` and floating point dtype.

    theta : numpy.ndarray (N-D array)
        The equivalent angular coordinate :math:`\theta_{eq}` with same shape and dtype as ``rho``.

    n : List[int]
        The radial orders.

    m : List[int]
        The azimuthal frequencies. Must have the same length as ``n``.

    x_derivative : List[int]
        The non-negative orders of the derivatives with respect to :math:`x`. Must have the same length as ``n``.

    y_derivative : List[int]
        The non-negative orders of the derivatives with respect to :math:`y`. Must have the same length as ``n``.

    Rx : float
        The semi-axis :math:`R_x` of the ellipse along its x axis. Must be strictly positive.

    Ry : float
        The semi-axis :math:`R_y` of the ellipse along its y axis. Must be strictly positive.

    alpha : float
        The rotation angle :math:`\alpha` of the ellipse in radians.

    h : float
        The ratio :math:`h` of the inner to the outer semi-axes, in :math:`[0, 1)`.

    precompute : bool
        If True, the powers of :math:`\rho_{eq}` and the cosine and sine terms are cached and shared
        between the polynomials.

    normalize : bool, optional
        If True, the polynomials (and their derivatives) are multiplied by the normalization factors of Noll
        :math:`N_{n}^{m} = \sqrt{(2 - \delta_{m0})(n + 1)}` (see :func:`pyzernike.core.core_normalization._core_normalization`).
        Default is False.

    Returns
    -------
    numpy.ndarray
        An array of shape ``(Q, ...)`` and dtype of ``rho``, where :math:`Q` is the length of ``n``.
        ``output[q]`` is the polynomial for the tuple ``(n[q], m[q], x_derivative[q], y_derivative[q])``.
        Invalid tuples (:math:`|m| > n` or :math:`n - m` odd) give zeros.

    """
    full_ellipse = h == 0.0
    s = 1.0 / (1.0 - h)
    inverse_r = None if full_ellipse else 1.0 / (h + (1.0 - h) * rho)
    cache = _XYTermsCache(rho, theta, precompute, inverse_r)

    output = numpy.zeros((len(n),) + rho.shape, dtype=rho.dtype)

    for idx in range(len(n)):
        order = x_derivative[idx] + y_derivative[idx]
        combined: XYFloatTerms = defaultdict(lambda: defaultdict(float))

        for u_order, weight in _uv_weights(x_derivative[idx], y_derivative[idx], Rx, Ry, alpha).items():
            for kind, k, radial in _xy_terms_uv_derivative(n[idx], m[idx], u_order, order - u_order):
                if full_ellipse:
                    radial = _xy_terms_merge_full_ellipse(radial)
                for (p, q), c in radial:
                    scale = 1.0 if full_ellipse else s ** (order - q)
                    combined[(kind, k)][(p, q)] += weight * 0.5 ** order * scale * c

        _xy_terms_evaluate(combined, cache, output[idx])

        if normalize:
            output[idx] *= _core_normalization([n[idx]], [m[idx]])[0]

    return output


def _core_xy_parameter_derivatives(
    rho: numpy.ndarray,
    theta: numpy.ndarray,
    n: List[int],
    m: List[int],
    parameter: List[str],
    Rx: float,
    Ry: float,
    alpha: float,
    h: float,
    precompute: bool,
    normalize: bool = False,
) -> numpy.ndarray:
    r"""
    Compute the first order derivatives of the Zernike polynomials on an elliptic annulus with respect to
    the parameters :math:`(x_0, y_0, R_x, R_y, \alpha, h)` of the annulus, at fixed :math:`(x, y)`.

    The polynomial depends on :math:`(x_0, y_0, R_x, R_y, \alpha)` only through :math:`u = X / R_x` and
    :math:`v = Y / R_y`, where :math:`X = c (x - x_0) + s (y - y_0)`, :math:`Y = -s (x - x_0) + c (y - y_0)`,
    :math:`c = \cos\alpha` and :math:`s = \sin\alpha`. With :math:`Z_u = \partial_u Z` and :math:`Z_v = \partial_v Z`:

    .. math::

        \frac{\partial Z}{\partial x_0} = -\frac{c}{R_x} Z_u + \frac{s}{R_y} Z_v,
        \quad
        \frac{\partial Z}{\partial y_0} = -\frac{s}{R_x} Z_u - \frac{c}{R_y} Z_v,
        \quad
        \frac{\partial Z}{\partial R_x} = -\frac{u}{R_x} Z_u,
        \quad
        \frac{\partial Z}{\partial R_y} = -\frac{v}{R_y} Z_v

    .. math::

        \frac{\partial Z}{\partial \alpha} = \frac{R_y}{R_x} v Z_u - \frac{R_x}{R_y} u Z_v

    The parameter :math:`h` appears only in :math:`\rho_{eq} = \frac{r - h}{1 - h}`, so at fixed :math:`(u, v)`:

    .. math::

        \frac{\partial Z}{\partial h} = \frac{\rho_{eq} - 1}{1 - h} \frac{\partial Z}{\partial \rho}(\rho_{eq}, \theta_{eq})

    :math:`Z_u` and :math:`Z_v` are computed with :func:`_core_xy_polynomials` for the ellipse
    :math:`R_x = R_y = 1`, :math:`\alpha = 0` (for which :math:`\partial_x = \partial_u` and :math:`\partial_y = \partial_v`),
    so they have no singularity at the center of the ellipse. :math:`\frac{\partial Z}{\partial \rho}` is computed with
    :func:`pyzernike.core.core_polynomial._core_polynomials`. The coordinates :math:`(u, v)` are recovered from
    :math:`u = r \cos\theta_{eq}` and :math:`v = r \sin\theta_{eq}` with :math:`r = h + (1 - h) \rho_{eq}`.

    Parameters
    ----------
    rho : numpy.ndarray (N-D array)
        The equivalent radial coordinate :math:`\rho_{eq}` of the points inside the elliptic annulus
        (:math:`0 \leq \rho_{eq} \leq 1`), with shape ``(...,)`` and floating point dtype.

    theta : numpy.ndarray (N-D array)
        The equivalent angular coordinate :math:`\theta_{eq}` with same shape and dtype as ``rho``.

    n : List[int]
        The radial orders.

    m : List[int]
        The azimuthal frequencies. Must have the same length as ``n``.

    parameter : List[str]
        The parameters with respect to which the polynomials are derived, in ``ELLIPSE_PARAMETERS``.
        Must have the same length as ``n``.

    Rx : float
        The semi-axis :math:`R_x` of the ellipse along its x axis. Must be strictly positive.

    Ry : float
        The semi-axis :math:`R_y` of the ellipse along its y axis. Must be strictly positive.

    alpha : float
        The rotation angle :math:`\alpha` of the ellipse in radians.

    h : float
        The ratio :math:`h` of the inner to the outer semi-axes, in :math:`[0, 1)`.

    precompute : bool
        If True, the powers of :math:`\rho_{eq}` and the cosine and sine terms are cached and shared
        between the polynomials.

    normalize : bool, optional
        If True, the derivatives are multiplied by the normalization factors of Noll
        :math:`N_{n}^{m} = \sqrt{(2 - \delta_{m0})(n + 1)}` (see :func:`pyzernike.core.core_normalization._core_normalization`).
        Default is False.

    Returns
    -------
    numpy.ndarray
        An array of shape ``(Q, ...)`` and dtype of ``rho``, where :math:`Q` is the length of ``n``.
        ``output[q]`` is the derivative of the polynomial ``(n[q], m[q])`` with respect to ``parameter[q]``.
        Invalid polynomials (:math:`|m| > n` or :math:`n - m` odd) give zeros.

    """
    cos_alpha, sin_alpha = math.cos(alpha), math.sin(alpha)

    # Z_u and Z_v of the polynomials derived with respect to (x0, y0, Rx, Ry, alpha)
    uv_keys = sorted({(n_i, m_i) for n_i, m_i, p in zip(n, m, parameter) if p != "h"})
    uv_values = _core_xy_polynomials(
        rho=rho,
        theta=theta,
        n=[n_i for n_i, _ in uv_keys] * 2,
        m=[m_i for _, m_i in uv_keys] * 2,
        x_derivative=[1] * len(uv_keys) + [0] * len(uv_keys),
        y_derivative=[0] * len(uv_keys) + [1] * len(uv_keys),
        Rx=1.0,
        Ry=1.0,
        alpha=0.0,
        h=h,
        precompute=precompute,
        normalize=False,
    )
    Z_u = dict(zip(uv_keys, uv_values[:len(uv_keys)]))
    Z_v = dict(zip(uv_keys, uv_values[len(uv_keys):]))

    # dZ/drho of the polynomials derived with respect to h
    h_keys = sorted({(n_i, m_i) for n_i, m_i, p in zip(n, m, parameter) if p == "h"})
    rho_values = _core_polynomials(
        rho=rho,
        theta=theta,
        n=[n_i for n_i, _ in h_keys],
        m=[m_i for _, m_i in h_keys],
        rho_derivative=[1] * len(h_keys),
        theta_derivative=[0] * len(h_keys),
        flag_radial=False,
        precompute=precompute,
        normalize=False,
    )
    Z_rho = dict(zip(h_keys, rho_values))

    if uv_keys:
        r = h + (1.0 - h) * rho
        u = r * numpy.cos(theta)
        v = r * numpy.sin(theta)

    output = numpy.empty((len(n),) + rho.shape, dtype=rho.dtype)

    for idx, (n_idx, m_idx, p) in enumerate(zip(n, m, parameter)):
        key = (n_idx, m_idx)
        if p == "x0":
            output[idx] = - (cos_alpha / Rx) * Z_u[key] + (sin_alpha / Ry) * Z_v[key]
        elif p == "y0":
            output[idx] = - (sin_alpha / Rx) * Z_u[key] - (cos_alpha / Ry) * Z_v[key]
        elif p == "Rx":
            output[idx] = - (u / Rx) * Z_u[key]
        elif p == "Ry":
            output[idx] = - (v / Ry) * Z_v[key]
        elif p == "alpha":
            output[idx] = (Ry / Rx) * v * Z_u[key] - (Rx / Ry) * u * Z_v[key]
        else:
            output[idx] = ((rho - 1.0) / (1.0 - h)) * Z_rho[key]

        if normalize:
            output[idx] *= _core_normalization([n_idx], [m_idx])[0]

    return output