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

from typing import Dict, List, Optional, Tuple

import sympy

from .core_polynomial import _radial_coefficients


RHO_SYMBOL = sympy.Symbol("r")
THETA_SYMBOL = sympy.Symbol("t")


def _build_precomputed_radial() -> Dict[Tuple[int, int], sympy.Expr]:
    r"""
    Build the table of the radial polynomials :math:`R_{n}^{m}(r)` for :math:`n \leq 19` and :math:`m \geq 0`.

    The table is built once when the module is imported. Sympy expressions are immutable, so the
    same objects can be safely returned to all the callers.

    Returns
    -------
    Dict[Tuple[int, int], sympy.Expr]
        The expressions indexed by ``(n, m)``.

    """
    r = RHO_SYMBOL
    return {
        (0, 0): sympy.Integer(1),
        (1, 1): r,
        (2, 0): 2*r**2 - 1,
        (2, 2): r**2,
        (3, 1): 3*r**3 - 2*r,
        (3, 3): r**3,
        (4, 0): 6*r**4 - 6*r**2 + 1,
        (4, 2): 4*r**4 - 3*r**2,
        (4, 4): r**4,
        (5, 1): 10*r**5 - 12*r**3 + 3*r,
        (5, 3): 5*r**5 - 4*r**3,
        (5, 5): r**5,
        (6, 0): 20*r**6 - 30*r**4 + 12*r**2 - 1,
        (6, 2): 15*r**6 - 20*r**4 + 6*r**2,
        (6, 4): 6*r**6 - 5*r**4,
        (6, 6): r**6,
        (7, 1): 35*r**7 - 60*r**5 + 30*r**3 - 4*r,
        (7, 3): 21*r**7 - 30*r**5 + 10*r**3,
        (7, 5): 7*r**7 - 6*r**5,
        (7, 7): r**7,
        (8, 0): 70*r**8 - 140*r**6 + 90*r**4 - 20*r**2 + 1,
        (8, 2): 56*r**8 - 105*r**6 + 60*r**4 - 10*r**2,
        (8, 4): 28*r**8 - 42*r**6 + 15*r**4,
        (8, 6): 8*r**8 - 7*r**6,
        (8, 8): r**8,
        (9, 1): 126*r**9 - 280*r**7 + 210*r**5 - 60*r**3 + 5*r,
        (9, 3): 84*r**9 - 168*r**7 + 105*r**5 - 20*r**3,
        (9, 5): 36*r**9 - 56*r**7 + 21*r**5,
        (9, 7): 9*r**9 - 8*r**7,
        (9, 9): r**9,
        (10, 0): 252*r**10 - 630*r**8 + 560*r**6 - 210*r**4 + 30*r**2 - 1,
        (10, 2): r**2*(210*r**8 - 504*r**6 + 420*r**4 - 140*r**2 + 15),
        (10, 4): r**4*(120*r**6 - 252*r**4 + 168*r**2 - 35),
        (10, 6): r**6*(45*r**4 - 72*r**2 + 28),
        (10, 8): r**8*(10*r**2 - 9),
        (10, 10): r**10,
        (11, 1): r*(462*r**10 - 1260*r**8 + 1260*r**6 - 560*r**4 + 105*r**2 - 6),
        (11, 3): r**3*(330*r**8 - 840*r**6 + 756*r**4 - 280*r**2 + 35),
        (11, 5): r**5*(165*r**6 - 360*r**4 + 252*r**2 - 56),
        (11, 7): r**7*(55*r**4 - 90*r**2 + 36),
        (11, 9): r**9*(11*r**2 - 10),
        (11, 11): r**11,
        (12, 0): 924*r**12 - 2772*r**10 + 3150*r**8 - 1680*r**6 + 420*r**4 - 42*r**2 + 1,
        (12, 2): r**2*(792*r**10 - 2310*r**8 + 2520*r**6 - 1260*r**4 + 280*r**2 - 21),
        (12, 4): r**4*(495*r**8 - 1320*r**6 + 1260*r**4 - 504*r**2 + 70),
        (12, 6): r**6*(220*r**6 - 495*r**4 + 360*r**2 - 84),
        (12, 8): r**8*(66*r**4 - 110*r**2 + 45),
        (12, 10): r**10*(12*r**2 - 11),
        (12, 12): r**12,
        (13, 1): r*(1716*r**12 - 5544*r**10 + 6930*r**8 - 4200*r**6 + 1260*r**4 - 168*r**2 + 7),
        (13, 3): r**3*(1287*r**10 - 3960*r**8 + 4620*r**6 - 2520*r**4 + 630*r**2 - 56),
        (13, 5): r**5*(715*r**8 - 1980*r**6 + 1980*r**4 - 840*r**2 + 126),
        (13, 7): r**7*(286*r**6 - 660*r**4 + 495*r**2 - 120),
        (13, 9): r**9*(78*r**4 - 132*r**2 + 55),
        (13, 11): r**11*(13*r**2 - 12),
        (13, 13): r**13,
        (14, 0): 3432*r**14 - 12012*r**12 + 16632*r**10 - 11550*r**8 + 4200*r**6 - 756*r**4 + 56*r**2 - 1,
        (14, 2): r**2*(3003*r**12 - 10296*r**10 + 13860*r**8 - 9240*r**6 + 3150*r**4 - 504*r**2 + 28),
        (14, 4): r**4*(2002*r**10 - 6435*r**8 + 7920*r**6 - 4620*r**4 + 1260*r**2 - 126),
        (14, 6): r**6*(1001*r**8 - 2860*r**6 + 2970*r**4 - 1320*r**2 + 210),
        (14, 8): r**8*(364*r**6 - 858*r**4 + 660*r**2 - 165),
        (14, 10): r**10*(91*r**4 - 156*r**2 + 66),
        (14, 12): r**12*(14*r**2 - 13),
        (14, 14): r**14,
        (15, 1): r*(6435*r**14 - 24024*r**12 + 36036*r**10 - 27720*r**8 + 11550*r**6 - 2520*r**4 + 252*r**2 - 8),
        (15, 3): r**3*(5005*r**12 - 18018*r**10 + 25740*r**8 - 18480*r**6 + 6930*r**4 - 1260*r**2 + 84),
        (15, 5): r**5*(3003*r**10 - 10010*r**8 + 12870*r**6 - 7920*r**4 + 2310*r**2 - 252),
        (15, 7): r**7*(1365*r**8 - 4004*r**6 + 4290*r**4 - 1980*r**2 + 330),
        (15, 9): r**9*(455*r**6 - 1092*r**4 + 858*r**2 - 220),
        (15, 11): r**11*(105*r**4 - 182*r**2 + 78),
        (15, 13): r**13*(15*r**2 - 14),
        (15, 15): r**15,
        (16, 0): 12870*r**16 - 51480*r**14 + 84084*r**12 - 72072*r**10 + 34650*r**8 - 9240*r**6 + 1260*r**4 - 72*r**2 + 1,
        (16, 2): r**2*(11440*r**14 - 45045*r**12 + 72072*r**10 - 60060*r**8 + 27720*r**6 - 6930*r**4 + 840*r**2 - 36),
        (16, 4): r**4*(8008*r**12 - 30030*r**10 + 45045*r**8 - 34320*r**6 + 13860*r**4 - 2772*r**2 + 210),
        (16, 6): r**6*(4368*r**10 - 15015*r**8 + 20020*r**6 - 12870*r**4 + 3960*r**2 - 462),
        (16, 8): r**8*(1820*r**8 - 5460*r**6 + 6006*r**4 - 2860*r**2 + 495),
        (16, 10): r**10*(560*r**6 - 1365*r**4 + 1092*r**2 - 286),
        (16, 12): r**12*(120*r**4 - 210*r**2 + 91),
        (16, 14): r**14*(16*r**2 - 15),
        (16, 16): r**16,
        (17, 1): r*(24310*r**16 - 102960*r**14 + 180180*r**12 - 168168*r**10 + 90090*r**8 - 27720*r**6 + 4620*r**4 - 360*r**2 + 9),
        (17, 3): r**3*(19448*r**14 - 80080*r**12 + 135135*r**10 - 120120*r**8 + 60060*r**6 - 16632*r**4 + 2310*r**2 - 120),
        (17, 5): r**5*(12376*r**12 - 48048*r**10 + 75075*r**8 - 60060*r**6 + 25740*r**4 - 5544*r**2 + 462),
        (17, 7): r**7*(6188*r**10 - 21840*r**8 + 30030*r**6 - 20020*r**4 + 6435*r**2 - 792),
        (17, 9): r**9*(2380*r**8 - 7280*r**6 + 8190*r**4 - 4004*r**2 + 715),
        (17, 11): r**11*(680*r**6 - 1680*r**4 + 1365*r**2 - 364),
        (17, 13): r**13*(136*r**4 - 240*r**2 + 105),
        (17, 15): r**15*(17*r**2 - 16),
        (17, 17): r**17,
        (18, 0): 48620*r**18 - 218790*r**16 + 411840*r**14 - 420420*r**12 + 252252*r**10 - 90090*r**8 + 18480*r**6 - 1980*r**4 + 90*r**2 - 1,
        (18, 2): r**2*(43758*r**16 - 194480*r**14 + 360360*r**12 - 360360*r**10 + 210210*r**8 - 72072*r**6 + 13860*r**4 - 1320*r**2 + 45),
        (18, 4): r**4*(31824*r**14 - 136136*r**12 + 240240*r**10 - 225225*r**8 + 120120*r**6 - 36036*r**4 + 5544*r**2 - 330),
        (18, 6): r**6*(18564*r**12 - 74256*r**10 + 120120*r**8 - 100100*r**6 + 45045*r**4 - 10296*r**2 + 924),
        (18, 8): r**8*(8568*r**10 - 30940*r**8 + 43680*r**6 - 30030*r**4 + 10010*r**2 - 1287),
        (18, 10): r**10*(3060*r**8 - 9520*r**6 + 10920*r**4 - 5460*r**2 + 1001),
        (18, 12): r**12*(816*r**6 - 2040*r**4 + 1680*r**2 - 455),
        (18, 14): r**14*(153*r**4 - 272*r**2 + 120),
        (18, 16): r**16*(18*r**2 - 17),
        (18, 18): r**18,
        (19, 1): r*(92378*r**18 - 437580*r**16 + 875160*r**14 - 960960*r**12 + 630630*r**10 - 252252*r**8 + 60060*r**6 - 7920*r**4 + 495*r**2 - 10),
        (19, 3): r**3*(75582*r**16 - 350064*r**14 + 680680*r**12 - 720720*r**10 + 450450*r**8 - 168168*r**6 + 36036*r**4 - 3960*r**2 + 165),
        (19, 5): r**5*(50388*r**14 - 222768*r**12 + 408408*r**10 - 400400*r**8 + 225225*r**6 - 72072*r**4 + 12012*r**2 - 792),
        (19, 7): r**7*(27132*r**12 - 111384*r**10 + 185640*r**8 - 160160*r**6 + 75075*r**4 - 18018*r**2 + 1716),
        (19, 9): r**9*(11628*r**10 - 42840*r**8 + 61880*r**6 - 43680*r**4 + 15015*r**2 - 2002),
        (19, 11): r**11*(3876*r**8 - 12240*r**6 + 14280*r**4 - 7280*r**2 + 1365),
        (19, 13): r**13*(969*r**6 - 2448*r**4 + 2040*r**2 - 560),
        (19, 15): r**15*(171*r**4 - 306*r**2 + 136),
        (19, 17): r**17*(19*r**2 - 18),
        (19, 19): r**19,
    }

_PRECOMPUTED_RADIAL = _build_precomputed_radial()

def _symbolic_precomputed(n: int, abs_m: int) -> Optional[sympy.Expr]:
    r"""
    Return the precomputed expression of the radial polynomial :math:`R_{n}^{m}(r)` if available.

    The table covers :math:`n \leq 19` without derivative, which makes the most common low orders
    a simple dictionary lookup. The expressions for :math:`n \geq 10` are kept in the factorized
    form :math:`r^{|m|} P(r^2)`.

    Parameters
    ----------
    n : int
        The radial order.

    abs_m : int
        The absolute value of the azimuthal frequency.

    Returns
    -------
    Optional[sympy.Expr]
        The expression in the symbol ``r``, or None if ``(n, abs_m)`` is not in the table.

    """
    return _PRECOMPUTED_RADIAL.get((n, abs_m))


def _is_valid_tuple(n: int, m: int, rho_derivative: int, theta_derivative: int, flag_radial: bool) -> bool:
    r"""
    Tell whether the tuple ``(n, m, rho_derivative, theta_derivative)`` gives a non-zero polynomial.

    The polynomial is identically zero if :math:`|m| > n`, if :math:`n - m` is odd, if
    :math:`a > n` (derivative of order greater than the degree), if :math:`m < 0` for a radial
    polynomial, or if :math:`m = 0` and :math:`b > 0` (angular derivative of a constant).

    Parameters
    ----------
    n : int
        The radial order.

    m : int
        The azimuthal frequency.

    rho_derivative : int
        The order :math:`a` of the derivative with respect to rho.

    theta_derivative : int
        The order :math:`b` of the derivative with respect to theta (ignored if ``flag_radial`` is True).

    flag_radial : bool
        If True, the tuple refers to a radial polynomial.

    Returns
    -------
    bool
        True if the polynomial is not identically zero.

    """
    if abs(m) > n or (n - m) % 2 != 0 or rho_derivative > n:
        return False
    if flag_radial:
        return m >= 0
    return not (m == 0 and theta_derivative > 0)


def _symbolic_radial(n: int, abs_m: int, rho_derivative: int) -> sympy.Expr:
    r"""
    Build the symbolic expression of the :math:`a`-th derivative of the radial polynomial.

    .. math::

        \frac{d^a R_{n}^{m}}{d\rho^a}(r) = \sum_{k=0}^{K} c_k \, r^{n-2k-a}
        \quad \text{with} \quad
        c_k = (-1)^k \frac{(n-k)!}{k! \left(\frac{n+|m|}{2}-k\right)! \left(\frac{n-|m|}{2}-k\right)!} \frac{(n-2k)!}{(n-2k-a)!}

    where :math:`K` is the largest :math:`k \leq \frac{n-|m|}{2}` such that :math:`n - 2k \geq a`.
    The integer coefficients :math:`c_k` are computed exactly, so the expression is built directly
    in expanded form, without any symbolic summation, simplification or differentiation.

    Parameters
    ----------
    n : int
        The radial order.

    abs_m : int
        The absolute value of the azimuthal frequency. Must satisfy :math:`|m| \leq n` and :math:`n - |m|` even.

    rho_derivative : int
        The order :math:`a` of the derivative with respect to rho. Must satisfy :math:`a \leq n`.

    Returns
    -------
    sympy.Expr
        The expanded polynomial in the symbol ``r``.

    """
    coefficients = _radial_coefficients(n, abs_m, rho_derivative)
    return sympy.Add(*[
        sympy.Integer(c) * RHO_SYMBOL ** (n - 2 * k - rho_derivative)
        for k, c in enumerate(coefficients)
    ])


def _symbolic_angular(m: int, theta_derivative: int) -> sympy.Expr:
    r"""
    Build the symbolic expression of the :math:`b`-th derivative of the angular part.

    .. math::

        \frac{d^b}{d\theta^b} \begin{cases}
            \cos(m t) & \text{if } m > 0 \\
            1 & \text{if } m = 0 \\
            \sin(|m| t) & \text{if } m < 0
        \end{cases}
        = \begin{cases}
            m^{b} \cos\left(m t + b\frac{\pi}{2}\right) & \text{if } m > 0 \\
            \delta_{b,0} & \text{if } m = 0 \\
            |m|^{b} \sin\left(|m| t + b\frac{\pi}{2}\right) & \text{if } m < 0
        \end{cases}

    The phase shift :math:`b\frac{\pi}{2}` is applied exactly by permuting and negating the cosine
    and sine according to :math:`b \bmod 4`, so that the expression only contains ``cos(|m| t)``
    or ``sin(|m| t)``.

    Parameters
    ----------
    m : int
        The azimuthal frequency.

    theta_derivative : int
        The order :math:`b` of the derivative with respect to theta.

    Returns
    -------
    sympy.Expr
        The angular factor in the symbol ``t``.

    """
    if m == 0:
        return sympy.Integer(1) if theta_derivative == 0 else sympy.Integer(0)

    abs_m = abs(m)
    cosine = sympy.cos(abs_m * THETA_SYMBOL)
    sine = sympy.sin(abs_m * THETA_SYMBOL)
    if m > 0:
        phases = (cosine, -sine, -cosine, sine)
    else:
        phases = (sine, cosine, -sine, -cosine)
    return sympy.Integer(abs_m) ** theta_derivative * phases[theta_derivative % 4]


def _core_symbolic(
        n: List[int],
        m: List[int],
        rho_derivative: List[int],
        theta_derivative: Optional[List[int]],
        flag_radial: bool,
    ) -> List[sympy.Expr]:
    r"""
    Compute the symbolic expressions of the Zernike polynomials :math:`Z_{n}^{m}(\rho, \theta)`
    (or of the radial polynomials :math:`R_{n}^{m}(\rho)` if ``flag_radial`` is True) and of
    their derivatives for each given tuple ``(n, m, rho_derivative, theta_derivative)``.

    The expressions use the symbol ``r`` for :math:`\rho` and ``t`` for :math:`\theta`, so that
    they can be evaluated with ``sympy.lambdify('r', expression)`` (radial) or
    ``sympy.lambdify(['r', 't'], expression)`` (full Zernike).

    Each expression is the product of the radial part and, if
    ``flag_radial`` is False, of the angular part (see :func:`_symbolic_angular`):

    .. math::

        \frac{\partial^{a+b} Z_{n}^{m}}{\partial \rho^{a} \partial \theta^{b}} = \begin{cases}
            \frac{d^a R_{n}^{m}}{d\rho^a} \, m^{b} \cos\left(m\theta + b\frac{\pi}{2}\right) & \text{if } m > 0 \\
            \frac{d^a R_{n}^{0}}{d\rho^a} \, \delta_{b,0} & \text{if } m = 0 \\
            \frac{d^a R_{n}^{|m|}}{d\rho^a} \, |m|^{b} \sin\left(|m|\theta + b\frac{\pi}{2}\right) & \text{if } m < 0
        \end{cases}

    The radial part is read from the precomputed table for :math:`n \leq 19` without derivative
    (see :func:`_symbolic_precomputed`), and built from the exact integer coefficients otherwise
    (see :func:`_symbolic_radial`). The coefficients are the same exact integers as in
    :func:`pyzernike.core._core_polynomials`, so both functions describe exactly the same polynomials.

    Parameters
    ----------
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

    Returns
    -------
    List[sympy.Expr]
        The symbolic expressions for each tuple ``(n, m, rho_derivative, theta_derivative)``.
        Invalid tuples (:math:`|m| > n`, :math:`n - m` odd,
        :math:`a > n`, or :math:`m < 0` if ``flag_radial`` is True) give ``sympy.Integer(0)``.

    """
    output: List[sympy.Expr] = []

    for idx in range(len(n)):
        n_idx = n[idx]
        m_idx = m[idx]
        a = rho_derivative[idx]
        b = 0 if flag_radial else theta_derivative[idx]

        if not _is_valid_tuple(n_idx, m_idx, a, b, flag_radial):
            output.append(sympy.Integer(0))
            continue

        expression = _symbolic_precomputed(n_idx, abs(m_idx)) if a == 0 else None
        if expression is None:
            expression = _symbolic_radial(n_idx, abs(m_idx), a)
        if not flag_radial:
            expression = expression * _symbolic_angular(m_idx, b)

        output.append(expression)

    return output