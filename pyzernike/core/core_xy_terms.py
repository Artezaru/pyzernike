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
Exact representation of the derivatives of the Zernike polynomials with respect to the Cartesian coordinates
:math:`(u, v)`, and their numerical evaluation (used by :func:`pyzernike.core.core_xy_polynomial._core_xy_polynomials`).

A Zernike polynomial and all its derivatives with respect to :math:`u = r \cos\theta` and :math:`v = r \sin\theta`
are written as sums of **terms**:

.. math::

    c \, \rho^{p} \, r^{-q} \, \cos(k\theta) \quad \text{or} \quad c \, \rho^{p} \, r^{-q} \, \sin(k\theta)

with :math:`k \geq 0`. The module has two parts:

1. **Exact algebra.** A term is stored as ``(kind, k, radial)`` where ``kind`` is ``"cos"`` or ``"sin"`` and
   ``radial`` is a tuple of monomials ``((p, q), c)`` with exact integer coefficients :math:`c`. The differential
   operators are transformations of these tuples, cached with ``functools.lru_cache``.
2. **Numerical evaluation.** The terms, combined with floating point coefficients (``XYFloatTerms``), are evaluated
   on the points with the Horner scheme, with the powers and the angular terms shared between the polynomials
   (see :class:`_XYTermsCache` and :func:`_xy_terms_evaluate`).
"""

from collections import defaultdict
from functools import lru_cache
from typing import Dict, Optional, Tuple

import numpy

from .core_polynomial import _radial_coefficients

Monomial = Tuple[int, int]
Radial = Tuple[Tuple[Monomial, int], ...]
Term = Tuple[str, int, Radial]
Terms = Tuple[Term, ...]

_XYTermsAccumulator = Dict[Tuple[str, int], Dict[Monomial, int]]

# Rules of 2 d_u and 2 d_v: (kind, variable) -> ((new kind, shift of k, sign, operator), ...)
# with the operators P = A' + (k/r) A and M = A' - (k/r) A (see _xy_terms_derive_uv).
_XY_TERMS_UV_RULES = {
    ("cos", "u"): (("cos", -1, 1, "P"), ("cos", 1, 1, "M")),
    ("cos", "v"): (("sin", 1, 1, "M"), ("sin", -1, -1, "P")),
    ("sin", "u"): (("sin", 1, 1, "M"), ("sin", -1, 1, "P")),
    ("sin", "v"): (("cos", -1, 1, "P"), ("cos", 1, -1, "M")),
}


def _xy_terms_accumulate(accumulator: _XYTermsAccumulator, kind: str, k: int, factor: int, radial: Dict[Monomial, int]) -> None:
    r"""
    Add ``factor * radial * kind(k theta)`` to ``accumulator``, with :math:`\cos(-k\theta) = \cos(k\theta)`,
    :math:`\sin(-k\theta) = -\sin(k\theta)` and :math:`\sin(0) = 0`.

    Parameters
    ----------
    accumulator : Dict[Tuple[str, int], Dict[Tuple[int, int], int]]
        The sum being built, by ``(kind, k)`` then by monomial ``(p, q)``. Modified in place.

    kind : str
        ``"cos"`` or ``"sin"``.

    k : int
        The angular frequency (may be negative).

    factor : int
        The integer factor applied to the radial part.

    radial : Dict[Tuple[int, int], int]
        The monomials ``(p, q) -> c`` of the radial part.

    """
    if k < 0:
        k = -k
        if kind == "sin":
            factor = -factor
    if factor == 0 or (kind == "sin" and k == 0):
        return
    target = accumulator.setdefault((kind, k), defaultdict(int))
    for monomial, coefficient in radial.items():
        target[monomial] += factor * coefficient


def _xy_terms_freeze(accumulator: _XYTermsAccumulator) -> Terms:
    r"""
    Convert an accumulator into sorted terms, without zero coefficients (hashable, for the caches).

    Parameters
    ----------
    accumulator : Dict[Tuple[str, int], Dict[Tuple[int, int], int]]
        The sum, by ``(kind, k)`` then by monomial ``(p, q)``.

    Returns
    -------
    Terms
        The terms ``(kind, k, radial)``.

    """
    terms = []
    for (kind, k), radial in sorted(accumulator.items()):
        monomials = tuple(sorted((monomial, c) for monomial, c in radial.items() if c != 0))
        if monomials:
            terms.append((kind, k, monomials))
    return tuple(terms)


def _xy_terms_radial_operator(radial: Radial, k: int) -> Dict[Monomial, int]:
    r"""
    Apply :math:`A \mapsto A' + \frac{k}{r} A` (where :math:`A' = \frac{dA}{dr}`) to a radial part written in
    the basis :math:`s^{D-q} \rho_{eq}^{p} r^{-q}`, where :math:`\rho_{eq} = \frac{r - h}{1 - h}`,
    :math:`s = \frac{1}{1-h} = \frac{d\rho_{eq}}{dr}` and :math:`D` is the total order of derivation:

    .. math::

        \frac{d}{dr} \left( s^{D-q} \rho_{eq}^{p} r^{-q} \right) = p \, s^{(D+1)-q} \rho_{eq}^{p-1} r^{-q} - q \, s^{(D+1)-(q+1)} \rho_{eq}^{p} r^{-(q+1)}

    and :math:`\frac{1}{r} s^{D-q} \rho_{eq}^{p} r^{-q} = s^{(D+1)-(q+1)} \rho_{eq}^{p} r^{-(q+1)}`. The integer
    coefficients therefore do not depend on :math:`h`. With :math:`-k`, the operator is :math:`A' - \frac{k}{r} A`.

    Parameters
    ----------
    radial : Radial
        The monomials ``((p, q), c)`` of the radial part :math:`A`.

    k : int
        The (signed) factor of :math:`\frac{A}{r}`.

    Returns
    -------
    Dict[Tuple[int, int], int]
        The monomials of :math:`A' + \frac{k}{r} A`.

    """
    output: Dict[Monomial, int] = defaultdict(int)
    for (p, q), c in radial:
        if p > 0:
            output[(p - 1, q)] += p * c
        output[(p, q + 1)] += (k - q) * c
    return output


def _xy_terms_derive_uv(terms: Terms, variable: str) -> Terms:
    r"""
    Apply :math:`2 \partial_u` or :math:`2 \partial_v`, where :math:`u = r \cos\theta` and :math:`v = r \sin\theta`.

    With :math:`P = A' + \frac{k}{r} A` and :math:`M = A' - \frac{k}{r} A` (see :func:`_xy_terms_radial_operator`):

    .. math::

        2 \partial_u [A \cos k\theta] = P \cos(k-1)\theta + M \cos(k+1)\theta

    .. math::

        2 \partial_v [A \cos k\theta] = M \sin(k+1)\theta - P \sin(k-1)\theta

    .. math::

        2 \partial_u [A \sin k\theta] = M \sin(k+1)\theta + P \sin(k-1)\theta

    .. math::

        2 \partial_v [A \sin k\theta] = P \cos(k-1)\theta - M \cos(k+1)\theta

    The factor :math:`\frac{1}{2}` is left out so that the coefficients stay integers.

    Parameters
    ----------
    terms : Terms
        The terms to derive.

    variable : str
        ``"u"`` or ``"v"``.

    Returns
    -------
    Terms
        The terms of the derivative, multiplied by 2.

    """
    accumulator: _XYTermsAccumulator = {}
    for kind, k, radial in terms:
        operators = {"P": _xy_terms_radial_operator(radial, k), "M": _xy_terms_radial_operator(radial, -k)}
        for new_kind, shift, sign, operator in _XY_TERMS_UV_RULES[(kind, variable)]:
            _xy_terms_accumulate(accumulator, new_kind, k + shift, sign, operators[operator])
    return _xy_terms_freeze(accumulator)


@lru_cache(maxsize=None)
def _xy_terms_zernike(n: int, m: int, flag_radial: bool) -> Terms:
    r"""
    Terms of :math:`Z_{n}^{m}(\rho, \theta)`, or of :math:`R_{n}^{m}(\rho)` if ``flag_radial`` is True
    (written as :math:`R_{n}^{m}(\rho) \cos(0 \theta)`).

    Parameters
    ----------
    n : int
        The radial order.

    m : int
        The azimuthal frequency.

    flag_radial : bool
        If True, the terms of the radial polynomial only.

    Returns
    -------
    Terms
        The terms. Empty if :math:`|m| > n`, :math:`n - m` is odd, or :math:`m < 0` with ``flag_radial``.

    """
    abs_m = abs(m)
    if abs_m > n or (n - m) % 2 != 0 or (flag_radial and m < 0):
        return ()
    radial = tuple(sorted(((n - 2 * j, 0), c) for j, c in enumerate(_radial_coefficients(n, abs_m, 0))))
    if flag_radial:
        return (("cos", 0, radial),)
    return (("cos" if m >= 0 else "sin", abs_m, radial),)


@lru_cache(maxsize=None)
def _xy_terms_uv_derivative(n: int, m: int, u_derivative: int, v_derivative: int) -> Terms:
    r"""
    Terms of :math:`2^{d} \partial_u^{d_u} \partial_v^{d_v} Z_{n}^{m}` with :math:`d = d_u + d_v`, in the basis
    :math:`s^{d-q} \rho_{eq}^{p} r^{-q}` (see :func:`_xy_terms_radial_operator`). They do not depend on the parameters
    of the ellipse. Cached.

    Parameters
    ----------
    n : int
        The radial order.

    m : int
        The azimuthal frequency.

    u_derivative : int
        The order :math:`d_u \geq 0` of the derivative with respect to :math:`u`.

    v_derivative : int
        The order :math:`d_v \geq 0` of the derivative with respect to :math:`v`.

    Returns
    -------
    Terms
        The terms of the derivative, multiplied by :math:`2^{d}`.

    """
    if u_derivative > 0:
        return _xy_terms_derive_uv(_xy_terms_uv_derivative(n, m, u_derivative - 1, v_derivative), "u")
    if v_derivative > 0:
        return _xy_terms_derive_uv(_xy_terms_uv_derivative(n, m, 0, v_derivative - 1), "v")
    return _xy_terms_zernike(n, m, False)


def _xy_terms_merge_full_ellipse(radial: Radial) -> Radial:
    r"""
    For :math:`h = 0`, :math:`\rho_{eq} = r` and :math:`s = 1`, so :math:`\rho_{eq}^{p} r^{-q} = r^{p-q}`.
    Merge the monomials with the same exponent :math:`e = p - q` (exactly, with integers).

    The negative exponents cancel exactly (the Zernike polynomials and their derivatives are polynomials
    in :math:`(u, v)`), so the result can be evaluated at :math:`r = 0`.

    Parameters
    ----------
    radial : Radial
        The monomials ``((p, q), c)`` of a radial part.

    Returns
    -------
    Radial
        The monomials ``((e, 0), c)`` with :math:`e \geq 0`.

    Raises
    ------
    RuntimeError
        If a negative exponent does not cancel (should never happen).

    """
    merged: Dict[int, int] = defaultdict(int)
    for (p, q), c in radial:
        merged[p - q] += c
    if any(e < 0 and c != 0 for e, c in merged.items()):
        raise RuntimeError("Internal error: a negative power of r does not cancel for h = 0.")
    return tuple(sorted(((e, 0), c) for e, c in merged.items() if c != 0))


# ============================================================================================================
# Numerical evaluation of the terms
# ============================================================================================================

# Sum of terms with floating point coefficients, by (kind, k) then by monomial (p, q)
XYFloatTerms = Dict[Tuple[str, int], Dict[Tuple[int, int], float]]


class _XYTermsCache:
    r"""
    The coordinates :math:`(\rho, \theta)` (and :math:`1/r`) of the evaluation points, with the powers of
    :math:`\rho` and the cosine and sine terms shared between the polynomials.

    Parameters
    ----------
    rho : numpy.ndarray (N-D array)
        The radial coordinate values with shape ``(...,)`` and floating point dtype.

    theta : numpy.ndarray (N-D array)
        The angular coordinate values with same shape and dtype as ``rho``.

    precompute : bool
        If True, the powers of :math:`\rho` and the cosine and sine terms are cached. If False, they are
        recomputed at each call (:math:`\rho^2` is always kept).

    inverse_r : Optional[numpy.ndarray] (N-D array), optional
        The values :math:`1/r` with same shape and dtype as ``rho``, required only to evaluate terms with
        :math:`q > 0`. Default is None.

    """

    def __init__(
        self,
        rho: numpy.ndarray,
        theta: numpy.ndarray,
        precompute: bool,
        inverse_r: Optional[numpy.ndarray] = None,
    ) -> None:
        self.rho = rho
        self.theta = theta
        self.precompute = precompute
        self.inverse_r = inverse_r
        self.rho_squared = rho * rho
        self._powers: Dict[int, numpy.ndarray] = {}
        self._angular: Dict[Tuple[str, int], numpy.ndarray] = {}

    def rho_power(self, exponent: int) -> numpy.ndarray:
        r"""
        Return :math:`\rho^{e}`.

        Parameters
        ----------
        exponent : int
            The positive exponent :math:`e`.

        Returns
        -------
        numpy.ndarray
            The values :math:`\rho^{e}` (must not be modified in place).

        """
        if exponent == 1:
            return self.rho
        if exponent == 2:
            return self.rho_squared
        if not self.precompute:
            return numpy.power(self.rho, exponent)
        if exponent not in self._powers:
            self._powers[exponent] = numpy.power(self.rho, exponent)
        return self._powers[exponent]

    def angular(self, kind: str, k: int) -> numpy.ndarray:
        r"""
        Return :math:`\cos(k\theta)` or :math:`\sin(k\theta)`.

        Parameters
        ----------
        kind : str
            ``"cos"`` or ``"sin"``.

        k : int
            The angular frequency :math:`k \geq 0`.

        Returns
        -------
        numpy.ndarray
            The values of the cosine or the sine (must not be modified in place).

        """
        function = numpy.cos if kind == "cos" else numpy.sin
        if not self.precompute:
            return function(k * self.theta)
        if (kind, k) not in self._angular:
            self._angular[(kind, k)] = function(k * self.theta)
        return self._angular[(kind, k)]


def _xy_terms_evaluate_radial(coefficients: Dict[int, float], cache: _XYTermsCache) -> numpy.ndarray:
    r"""
    Evaluate :math:`\sum_e c_e \rho^e` with the Horner scheme (in :math:`\rho^2` if all the exponents have
    the same parity):

    .. math::

        \sum_e c_e \rho^e = \rho^{e_{min}} \left( \left( c_{e_{max}} \rho^2 + c_{e_{max}-2} \right) \rho^2 + \dots + c_{e_{min}} \right)

    Parameters
    ----------
    coefficients : Dict[int, float]
        The non-empty mapping from the non-negative exponents :math:`e` to the coefficients :math:`c_e`.

    cache : _XYTermsCache
        The coordinates and the cached powers of :math:`\rho`.

    Returns
    -------
    numpy.ndarray
        A new array with the values of the polynomial, with the shape and dtype of ``cache.rho``.

    """
    exponents = sorted(coefficients)
    lowest, highest = exponents[0], exponents[-1]
    step = 2 if all((e - lowest) % 2 == 0 for e in exponents) else 1
    base = cache.rho_squared if step == 2 else cache.rho

    result = numpy.full(cache.rho.shape, coefficients[highest], dtype=cache.rho.dtype)
    for exponent in range(highest - step, lowest - 1, -step):
        result *= base
        coefficient = coefficients.get(exponent, 0.0)
        if coefficient != 0.0:
            result += coefficient
    if lowest > 0:
        result *= cache.rho_power(lowest)
    return result


def _xy_terms_evaluate(terms: XYFloatTerms, cache: _XYTermsCache, output: numpy.ndarray) -> None:
    r"""
    Add the sum of terms :math:`c \, \rho^{p} \, r^{-q} \, \cos(k\theta)` (or :math:`\sin`) to ``output``.

    For each ``(kind, k)``, the radial part is evaluated with the Horner scheme in :math:`1/r`, whose
    coefficients are polynomials in :math:`\rho` (see :func:`_xy_terms_evaluate_radial`):

    .. math::

        \sum_{p, q} c_{p,q} \rho^{p} r^{-q} = \sum_{q} r^{-q} P_q(\rho)

    Parameters
    ----------
    terms : Dict[Tuple[str, int], Dict[Tuple[int, int], float]]
        The coefficients by ``(kind, k)`` then by monomial ``(p, q)``.

    cache : _XYTermsCache
        The coordinates and the cached powers and angular terms. ``cache.inverse_r`` is required if a
        monomial has :math:`q > 0`.

    output : numpy.ndarray
        The array with the shape and dtype of ``cache.rho`` to which the sum is added (in place).

    """
    for (kind, k), monomials in terms.items():
        by_q: Dict[int, Dict[int, float]] = defaultdict(dict)
        for (p, q), c in monomials.items():
            if c != 0.0:
                by_q[q][p] = c
        if not by_q:
            continue

        radial = None
        for q in range(max(by_q), -1, -1):
            if radial is not None:
                radial *= cache.inverse_r
            if q in by_q:
                polynomial = _xy_terms_evaluate_radial(by_q[q], cache)
                radial = polynomial if radial is None else radial + polynomial

        if kind == "cos" and k == 0:
            output += radial
        else:
            output += radial * cache.angular(kind, k)