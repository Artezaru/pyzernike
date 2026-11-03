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
from typing import Dict, List, Optional

import numpy

from .core_normalization import _core_normalization


def _radial_coefficients(n: int, abs_m: int, rho_derivative: int) -> List[int]:
    r"""
    Exact integer coefficients of :math:`\frac{d^a R_{n}^{m}}{d\rho^a}` by decreasing power of :math:`\rho`:

    .. math::

        c_k = (-1)^k \binom{n-k}{k} \binom{n-2k}{\frac{n-|m|}{2}-k} \frac{(n-2k)!}{(n-2k-a)!}

    associated with :math:`\rho^{n-2k-a}`, for :math:`k = 0, 1, \dots` while :math:`n - 2k \geq a`.

    Parameters
    ----------
    n : int
        The radial order, with :math:`n \geq |m|` and :math:`n - |m|` even.

    abs_m : int
        The absolute value :math:`|m|` of the azimuthal frequency.

    rho_derivative : int
        The order :math:`a \geq 0` of the derivative with respect to :math:`\rho`.

    Returns
    -------
    List[int]
        The coefficients :math:`c_0, c_1, \dots`.

    """
    s = (n - abs_m) // 2
    coefficients = []
    for k in range(s + 1):
        exponent = n - 2 * k
        if exponent < rho_derivative:
            break
        c = math.comb(n - k, k) * math.comb(exponent, s - k) * math.perm(exponent, rho_derivative)
        coefficients.append(-c if k % 2 else c)
    return coefficients


def _core_polynomials(
        rho: numpy.ndarray,
        theta: Optional[numpy.ndarray],
        n: List[int],
        m: List[int],
        rho_derivative: List[int],
        theta_derivative: Optional[List[int]],
        flag_radial: bool,
        precompute: bool,
        normalize: bool,
    ) -> numpy.ndarray:
    r"""
    Compute the Zernike polynomials :math:`Z_{n}^{m}(\rho, \theta)` (or the radial polynomials
    :math:`R_{n}^{m}(\rho)` if ``flag_radial`` is True) and their derivatives for each given
    tuple ``(n, m, rho_derivative, theta_derivative)`` using the explicit formula.

    The :math:`a`-th derivative of the radial polynomial is:

    .. math::

        \frac{d^a R_{n}^{m}}{d\rho^a}(\rho) = \sum_{k=0}^{K} c_k \, \rho^{n-2k-a}
        \quad \text{with} \quad
        c_k = (-1)^k \frac{(n-k)!}{k! \left(\frac{n+|m|}{2}-k\right)! \left(\frac{n-|m|}{2}-k\right)!} \frac{(n-2k)!}{(n-2k-a)!}

    where :math:`K` is the largest :math:`k \leq \frac{n-|m|}{2}` such that :math:`n - 2k \geq a`.

    The coefficients are computed exactly as integers with ``math.comb`` and ``math.perm``
    and rounded once to the dtype of ``rho``. The sum is evaluated with the Horner scheme in :math:`\rho^2`:

    .. math::

        \frac{d^a R_{n}^{m}}{d\rho^a}(\rho) = \rho^{n-2K-a} \left( \left( c_0 \rho^2 + c_1 \right) \rho^2 + \dots + c_K \right)

    The full Zernike polynomials and their angular derivatives are then assembled as:

    .. math::

        \frac{\partial^{a+b} Z_{n}^{m}}{\partial \rho^{a} \partial \theta^{b}} = \begin{cases}
            \frac{d^a R_{n}^{m}}{d\rho^a} \, m^{b} \cos\left(m\theta + b\frac{\pi}{2}\right) & \text{if } m > 0 \\
            \frac{d^a R_{n}^{0}}{d\rho^a} \, \delta_{b,0} & \text{if } m = 0 \\
            \frac{d^a R_{n}^{|m|}}{d\rho^a} \, |m|^{b} \sin\left(|m|\theta + b\frac{\pi}{2}\right) & \text{if } m < 0
        \end{cases}

    .. note::

        The alternating sum suffers from cancellation for high orders (absolute error of the order of
        :math:`\epsilon \sum_k |c_k|`). In double precision, it stays below :math:`10^{-10}` up to
        :math:`n \approx 20` for :math:`m = 0`. For higher orders, prefer
        :func:`pyzernike.core._core_recursive_up_to_order`.

    Parameters
    ----------
    rho : numpy.ndarray (N-D array)
        The radial coordinate values with shape ``(...,)`` and floating point dtype.

    theta : Optional[numpy.ndarray] (N-D array)
        The angular coordinate values with same shape and dtype as ``rho``.
        Must be None if ``flag_radial`` is True.

    n : List[int]
        The radial orders.

    m : List[int]
        The azimuthal frequencies. Must have the same length as ``n``.

    rho_derivative : List[int]
        The orders of the derivatives with respect to rho. Must have the same length as ``n``.

    theta_derivative : Optional[List[int]]
        The orders of the derivatives with respect to theta. Must have the same length as ``n``.
        Must be None if ``flag_radial`` is True.

    flag_radial : bool
        If True, computes the radial polynomials only. If False, computes the full Zernike polynomials.

    precompute : bool
        If True, the powers of rho and the cosine and sine terms are cached and shared between the polynomials.

    normalize : bool
        If True and ``flag_radial`` is False, the Zernike polynomials (and their derivatives) are multiplied by the
        normalization factors of Noll :math:`N_{n}^{m} = \sqrt{(2 - \delta_{m0})(n + 1)}` (see
        :func:`pyzernike.core.core_normalization._core_normalization`). Ignored for the radial polynomials.

    Returns
    -------
    numpy.ndarray
        An array of shape ``(Q, ...)`` and dtype of ``rho``, where :math:`Q` is the length of ``n``.
        ``output[q]`` is the polynomial for the tuple ``(n[q], m[q], rho_derivative[q], theta_derivative[q])``.
        Invalid tuples (:math:`|m| > n`, :math:`n - m` odd, :math:`a > n`, or
        :math:`m < 0` if ``flag_radial`` is True) give zeros.

    """
    shape = rho.shape
    dtype = rho.dtype

    rho_squared = rho * rho
    powers_cache: Dict[int, numpy.ndarray] = {}
    angular_cache: Dict[tuple, numpy.ndarray] = {}

    def _rho_power(exponent: int) -> numpy.ndarray:
        if exponent == 1:
            return rho
        if exponent == 2:
            return rho_squared
        if not precompute:
            return numpy.power(rho, exponent)
        if exponent not in powers_cache:
            powers_cache[exponent] = numpy.power(rho, exponent)
        return powers_cache[exponent]

    def _angular(func: str, abs_m: int) -> numpy.ndarray:
        if not precompute:
            return numpy.cos(abs_m * theta) if func == "cos" else numpy.sin(abs_m * theta)
        key = (func, abs_m)
        if key not in angular_cache:
            angular_cache[key] = numpy.cos(abs_m * theta) if func == "cos" else numpy.sin(abs_m * theta)
        return angular_cache[key]

    COSINE_DERIVATIVE = {0: (1, "cos"), 1: (-1, "sin"), 2: (-1, "cos"), 3: (1, "sin")}
    SINE_DERIVATIVE = {0: (1, "sin"), 1: (1, "cos"), 2: (-1, "sin"), 3: (-1, "cos")}

    output = numpy.empty((len(n),) + shape, dtype=dtype)

    for idx in range(len(n)):
        n_idx = n[idx]
        m_idx = m[idx]
        abs_m_idx = abs(m_idx)
        a = rho_derivative[idx]
        b = 0 if flag_radial else theta_derivative[idx]

        if abs_m_idx > n_idx or (n_idx - m_idx) % 2 != 0 or a > n_idx \
                or (flag_radial and m_idx < 0) or (m_idx == 0 and b > 0):
            output[idx] = 0.0
            continue

        coefficients = _radial_coefficients(n_idx, abs_m_idx, a)
        lowest_exponent = n_idx - a - 2 * (len(coefficients) - 1)

        result = output[idx]
        result.fill(coefficients[0])
        for coefficient in coefficients[1:]:
            result *= rho_squared
            result += float(coefficient)
        if lowest_exponent > 0:
            result *= _rho_power(lowest_exponent)

        if not flag_radial and m_idx != 0:
            sign, func = (COSINE_DERIVATIVE if m_idx > 0 else SINE_DERIVATIVE)[b % 4]
            result *= _angular(func, abs_m_idx)
            factor = sign * abs_m_idx ** b
            if factor != 1:
                result *= float(factor)

        if normalize and not flag_radial:
            result *= _core_normalization([n_idx], [m_idx])[0]

    return output