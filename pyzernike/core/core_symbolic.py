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

from typing import List, Optional
import numpy
import sympy

def core_symbolic(
        n: numpy.array,
        m: numpy.array,
        rho_derivative: numpy.array,
        theta_derivative: Optional[numpy.array],
        flag_radial: bool,
    ) -> List[sympy.Expr]:
    r"""

    Compute the symbolic expression of the Zernike polynomial :math:`Z_{n}^{m}(\rho, \theta)` or the radial Zenike
    polynomial :math:`R_{n}^{m}(\rho)` if the flag `flag_radial` is set to True with symbolic ``sympy`` computation.

    .. warning::

        This method is a core function of ``pyzernike`` that is not designed to be use by the users directly.
        Please use the high level functions.

    .. seealso::

        - :func:`pyzernike.radial_symbolic` for the radial Zernike polynomial symbolic computation.
        - :func:`pyzernike.zernike_symbolic` for the full Zernike polynomial symbolic computation.
        - The page :doc:`../../mathematical_description` in the documentation for the mathematical description of the Zernike polynomials.

    - ``n``, ``m``, ``rho_derivative`` and ``theta_derivative`` are expected to be sequences of integers of the same length and valid values.

    .. note::

        The `r` symbol is used to represent the radial coordinate :math:`\rho` in the symbolic expression.
        The `t` symbol is used to represent the angular coordinate :math:`\theta` in the symbolic expression.

    .. warning::

        Note match the numpy ``core_polynomial`` function for n=11, m=9, dr=0, dt=10, to investigate later.

    Parameters
    ----------    
    n : numpy.array[numpy.integer]
        The orders of the Zernike polynomials to compute. Must be a 1D array of integers.

    m : numpy.array[numpy.integer]
        The azimuthal frequencies of the Zernike polynomials. Must be a 1D array of integers.

    numpy.array[numpy.integer]
        The orders of the derivatives with respect to rho. Must be a 1D array of integers.

    theta_derivative : Optional[numpy.array[numpy.integer]]
        The orders of the derivatives with respect to theta. Must be None if ``flag_radial`` is True. Otherwise, must be a 1D array of integers.

    flag_radial : bool
        If True, the radial Zernike polynomial :math:`R_{n}^{m}(\rho)` is computed instead of the full Zernike polynomial :math:`Z_{n}^{m}(\rho, \theta)`.
        If False, the full Zernike polynomial :math:`Z_{n}^{m}(\rho, \theta)` is computed, which includes the angular part with the cosine and sine terms.

    float_type : numpy.floating
        The floating point type used for the computations (e.g., numpy.float32, numpy.float64).

    Returns
    -------
    List[sympy.Expr]
        A list of symbolic expressions of the Zernike polynomial or radial Zernike polynomial for each order and azimuthal frequency specified in `n` and `m`.
        Each expression is a sympy expression that can be evaluated or manipulated further.
    """

    # Fast assertions on the inputs
    assert isinstance(n, numpy.ndarray) and n.ndim == 1 and numpy.issubdtype(n.dtype, numpy.integer), "[pyzernike-core] n must be a 1D numpy array of integers."
    assert isinstance(m, numpy.ndarray) and m.ndim == 1 and numpy.issubdtype(m.dtype, numpy.integer), "[pyzernike-core] m must be a 1D numpy array of integers."
    assert isinstance(rho_derivative, numpy.ndarray) and rho_derivative.ndim == 1 and numpy.issubdtype(rho_derivative.dtype, numpy.integer), "[pyzernike-core] rho_derivative must be a 1D numpy array of integers."
    assert isinstance(flag_radial, bool), "[pyzernike-core] flag_radial must be a boolean."
    assert flag_radial or (isinstance(theta_derivative, numpy.ndarray) and theta_derivative.ndim == 1 and numpy.issubdtype(theta_derivative.dtype, numpy.integer)), "[pyzernike-core] theta_derivative must be a 1D numpy array of integers when flag_radial is False."
    assert not flag_radial or theta_derivative is None, "[pyzernike-core] theta_derivative must be None when flag_radial is True."
    assert n.size == m.size == rho_derivative.size and (flag_radial or n.size == theta_derivative.size), "[pyzernike-core] n, m, rho_derivative and theta_derivative (if flag_radial is False) must have the same size."

    # Create the output list
    output: list[sympy.Expr] = []

    # =================================================================
    # Boucle over the polynomials to compute
    # =================================================================
    for idx in range(len(n)):

        # Extract the n, m and rho_derivative
        n_idx: numpy.integer = n[idx]
        m_idx: numpy.integer = m[idx]
        rho_derivative_idx: numpy.integer = rho_derivative[idx]

        # Construct the radial polynomial expression
        r = sympy.symbols('r')

        if n_idx < 0 or (n_idx - m_idx) % 2 != 0 or abs(m_idx) > n_idx:
            expression = sympy.sympify(0) * r # Need to lambdify to work correctly (must contain r)

        # Case for derivatives of order greater than n_idx
        elif rho_derivative_idx > n_idx:
            expression = sympy.sympify(0) * r

        # Case for radial polynomial only
        elif flag_radial and m_idx < 0:
            expression = sympy.sympify(0) * r

        # Compute the symbolic expression for the radial polynomial with derivative = 0
        elif n_idx == 0:
            if abs(m_idx) == 0:
                expression = sympy.sympify(1) + 0 * r

        elif n_idx == 1:
            if m_idx == 1:
                expression = r

        elif n_idx == 2:
            if m_idx == 0:
                expression = 2*r**2 - 1
            elif m_idx == 2:
                expression = r**2

        elif n_idx == 3:
            if m_idx == 1:
                expression = 3*r**3 - 2*r
            elif m_idx == 3:
                expression = r**3

        elif n_idx == 4:
            if m_idx == 0:
                expression = 6*r**4 - 6*r**2 + 1
            elif m_idx == 2:
                expression = 4*r**4 - 3*r**2
            elif m_idx == 4:
                expression = r**4

        elif n_idx == 5:
            if m_idx == 1:
                expression = 10*r**5 - 12*r**3 + 3*r
            elif m_idx == 3:
                expression = 5*r**5 - 4*r**3
            elif m_idx == 5:
                expression = r**5

        elif n_idx == 6:
            if m_idx == 0:
                expression = 20*r**6 - 30*r**4 + 12*r**2 - 1
            elif m_idx == 2:
                expression = 15*r**6 - 20*r**4 + 6*r**2
            elif m_idx == 4:
                expression = 6*r**6 - 5*r**4
            elif m_idx == 6:
                expression = r**6

        elif n_idx == 7:
            if m_idx == 1:
                expression = 35*r**7 - 60*r**5 + 30*r**3 - 4*r
            elif m_idx == 3:
                expression = 21*r**7 - 30*r**5 + 10*r**3
            elif m_idx == 5:
                expression = 7*r**7 - 6*r**5
            elif m_idx == 7:
                expression = r**7

        elif n_idx == 8:
            if m_idx == 0:
                expression = 70*r**8 - 140*r**6 + 90*r**4 - 20*r**2 + 1
            elif m_idx == 2:
                expression = 56*r**8 - 105*r**6 + 60*r**4 - 10*r**2
            elif m_idx == 4:
                expression = 28*r**8 - 42*r**6 + 15*r**4
            elif m_idx == 6:
                expression = 8*r**8 - 7*r**6
            elif m_idx == 8:
                expression = r**8

        elif n_idx == 9:
            if m_idx == 1:
                expression = 126*r**9 - 280*r**7 + 210*r**5 - 60*r**3 + 5*r
            elif m_idx == 3:
                expression = 84*r**9 - 168*r**7 + 105*r**5 - 20*r**3
            elif m_idx == 5:
                expression = 36*r**9 - 56*r**7 + 21*r**5
            elif m_idx == 7:
                expression = 9*r**9 - 8*r**7
            elif m_idx == 9:
                expression = r**9

        elif n_idx == 10:
            if m_idx == 0:
                expression = 252*r**10 - 630*r**8 + 560*r**6 - 210*r**4 + 30*r**2 - 1
            elif m_idx == 2:
                expression = r**2*(210*r**8 - 504*r**6 + 420*r**4 - 140*r**2 + 15)
            elif m_idx == 4:
                expression = r**4*(120*r**6 - 252*r**4 + 168*r**2 - 35)
            elif m_idx == 6:
                expression = r**6*(45*r**4 - 72*r**2 + 28)
            elif m_idx == 8:
                expression = r**8*(10*r**2 - 9)
            elif m_idx == 10:
                expression = r**10

        elif n_idx == 11:
            if m_idx == 1:
                expression = r*(462*r**10 - 1260*r**8 + 1260*r**6 - 560*r**4 + 105*r**2 - 6)
            elif m_idx == 3:
                expression = r**3*(330*r**8 - 840*r**6 + 756*r**4 - 280*r**2 + 35)
            elif m_idx == 5:
                expression = r**5*(165*r**6 - 360*r**4 + 252*r**2 - 56)
            elif m_idx == 7:
                expression = r**7*(55*r**4 - 90*r**2 + 36)
            elif m_idx == 9:
                expression = r**9*(11*r**2 - 10)
            elif m_idx == 11:
                expression = r**11

        elif n_idx == 12:
            if m_idx == 0:
                expression = 924*r**12 - 2772*r**10 + 3150*r**8 - 1680*r**6 + 420*r**4 - 42*r**2 + 1
            elif m_idx == 2:
                expression = r**2*(792*r**10 - 2310*r**8 + 2520*r**6 - 1260*r**4 + 280*r**2 - 21)
            elif m_idx == 4:
                expression = r**4*(495*r**8 - 1320*r**6 + 1260*r**4 - 504*r**2 + 70)
            elif m_idx == 6:
                expression = r**6*(220*r**6 - 495*r**4 + 360*r**2 - 84)
            elif m_idx == 8:
                expression = r**8*(66*r**4 - 110*r**2 + 45)
            elif m_idx == 10:
                expression = r**10*(12*r**2 - 11)
            elif m_idx == 12:
                expression = r**12

        elif n_idx == 13:
            if m_idx == 1:
                expression = r*(1716*r**12 - 5544*r**10 + 6930*r**8 - 4200*r**6 + 1260*r**4 - 168*r**2 + 7)
            elif m_idx == 3:
                expression = r**3*(1287*r**10 - 3960*r**8 + 4620*r**6 - 2520*r**4 + 630*r**2 - 56)
            elif m_idx == 5:
                expression = r**5*(715*r**8 - 1980*r**6 + 1980*r**4 - 840*r**2 + 126)
            elif m_idx == 7:
                expression = r**7*(286*r**6 - 660*r**4 + 495*r**2 - 120)
            elif m_idx == 9:
                expression = r**9*(78*r**4 - 132*r**2 + 55)
            elif m_idx == 11:
                expression = r**11*(13*r**2 - 12)
            elif m_idx == 13:
                expression = r**13

        elif n_idx == 14:
            if m_idx == 0:
                expression = 3432*r**14 - 12012*r**12 + 16632*r**10 - 11550*r**8 + 4200*r**6 - 756*r**4 + 56*r**2 - 1
            elif m_idx == 2:
                expression = r**2*(3003*r**12 - 10296*r**10 + 13860*r**8 - 9240*r**6 + 3150*r**4 - 504*r**2 + 28)
            elif m_idx == 4:
                expression = r**4*(2002*r**10 - 6435*r**8 + 7920*r**6 - 4620*r**4 + 1260*r**2 - 126)
            elif m_idx == 6:
                expression = r**6*(1001*r**8 - 2860*r**6 + 2970*r**4 - 1320*r**2 + 210)
            elif m_idx == 8:
                expression = r**8*(364*r**6 - 858*r**4 + 660*r**2 - 165)
            elif m_idx == 10:
                expression = r**10*(91*r**4 - 156*r**2 + 66)
            elif m_idx == 12:
                expression = r**12*(14*r**2 - 13)
            elif m_idx == 14:
                expression = r**14

        elif n_idx == 15:
            if m_idx == 1:
                expression = r*(6435*r**14 - 24024*r**12 + 36036*r**10 - 27720*r**8 + 11550*r**6 - 2520*r**4 + 252*r**2 - 8)
            elif m_idx == 3:
                expression = r**3*(5005*r**12 - 18018*r**10 + 25740*r**8 - 18480*r**6 + 6930*r**4 - 1260*r**2 + 84)
            elif m_idx == 5:
                expression = r**5*(3003*r**10 - 10010*r**8 + 12870*r**6 - 7920*r**4 + 2310*r**2 - 252)
            elif m_idx == 7:
                expression = r**7*(1365*r**8 - 4004*r**6 + 4290*r**4 - 1980*r**2 + 330)
            elif m_idx == 9:
                expression = r**9*(455*r**6 - 1092*r**4 + 858*r**2 - 220)
            elif m_idx == 11:
                expression = r**11*(105*r**4 - 182*r**2 + 78)
            elif m_idx == 13:
                expression = r**13*(15*r**2 - 14)
            elif m_idx == 15:
                expression = r**15

        elif n_idx == 16:
            if m_idx == 0:
                expression = 12870*r**16 - 51480*r**14 + 84084*r**12 - 72072*r**10 + 34650*r**8 - 9240*r**6 + 1260*r**4 - 72*r**2 + 1
            elif m_idx == 2:
                expression = r**2*(11440*r**14 - 45045*r**12 + 72072*r**10 - 60060*r**8 + 27720*r**6 - 6930*r**4 + 840*r**2 - 36)
            elif m_idx == 4:
                expression = r**4*(8008*r**12 - 30030*r**10 + 45045*r**8 - 34320*r**6 + 13860*r**4 - 2772*r**2 + 210)
            elif m_idx == 6:
                expression = r**6*(4368*r**10 - 15015*r**8 + 20020*r**6 - 12870*r**4 + 3960*r**2 - 462)
            elif m_idx == 8:
                expression = r**8*(1820*r**8 - 5460*r**6 + 6006*r**4 - 2860*r**2 + 495)
            elif m_idx == 10:
                expression = r**10*(560*r**6 - 1365*r**4 + 1092*r**2 - 286)
            elif m_idx == 12:
                expression = r**12*(120*r**4 - 210*r**2 + 91)
            elif m_idx == 14:
                expression = r**14*(16*r**2 - 15)
            elif m_idx == 16:
                expression = r**16

        elif n_idx == 17:
            if m_idx == 1:
                expression = r*(24310*r**16 - 102960*r**14 + 180180*r**12 - 168168*r**10 + 90090*r**8 - 27720*r**6 + 4620*r**4 - 360*r**2 + 9)
            elif m_idx == 3:
                expression = r**3*(19448*r**14 - 80080*r**12 + 135135*r**10 - 120120*r**8 + 60060*r**6 - 16632*r**4 + 2310*r**2 - 120)
            elif m_idx == 5:
                expression = r**5*(12376*r**12 - 48048*r**10 + 75075*r**8 - 60060*r**6 + 25740*r**4 - 5544*r**2 + 462)
            elif m_idx == 7:
                expression = r**7*(6188*r**10 - 21840*r**8 + 30030*r**6 - 20020*r**4 + 6435*r**2 - 792)
            elif m_idx == 9:
                expression = r**9*(2380*r**8 - 7280*r**6 + 8190*r**4 - 4004*r**2 + 715)
            elif m_idx == 11:
                expression = r**11*(680*r**6 - 1680*r**4 + 1365*r**2 - 364)
            elif m_idx == 13:
                expression = r**13*(136*r**4 - 240*r**2 + 105)
            elif m_idx == 15:
                expression = r**15*(17*r**2 - 16)
            elif m_idx == 17:
                expression = r**17

        elif n_idx == 18:
            if m_idx == 0:
                expression = 48620*r**18 - 218790*r**16 + 411840*r**14 - 420420*r**12 + 252252*r**10 - 90090*r**8 + 18480*r**6 - 1980*r**4 + 90*r**2 - 1
            elif m_idx == 2:
                expression = r**2*(43758*r**16 - 194480*r**14 + 360360*r**12 - 360360*r**10 + 210210*r**8 - 72072*r**6 + 13860*r**4 - 1320*r**2 + 45)
            elif m_idx == 4:
                expression = r**4*(31824*r**14 - 136136*r**12 + 240240*r**10 - 225225*r**8 + 120120*r**6 - 36036*r**4 + 5544*r**2 - 330)
            elif m_idx == 6:
                expression = r**6*(18564*r**12 - 74256*r**10 + 120120*r**8 - 100100*r**6 + 45045*r**4 - 10296*r**2 + 924)
            elif m_idx == 8:
                expression = r**8*(8568*r**10 - 30940*r**8 + 43680*r**6 - 30030*r**4 + 10010*r**2 - 1287)
            elif m_idx == 10:
                expression = r**10*(3060*r**8 - 9520*r**6 + 10920*r**4 - 5460*r**2 + 1001)
            elif m_idx == 12:
                expression = r**12*(816*r**6 - 2040*r**4 + 1680*r**2 - 455)
            elif m_idx == 14:
                expression = r**14*(153*r**4 - 272*r**2 + 120)
            elif m_idx == 16:
                expression = r**16*(18*r**2 - 17)
            elif m_idx == 18:
                expression = r**18

        elif n_idx == 19:
            if m_idx == 1:
                expression = r*(92378*r**18 - 437580*r**16 + 875160*r**14 - 960960*r**12 + 630630*r**10 - 252252*r**8 + 60060*r**6 - 7920*r**4 + 495*r**2 - 10)
            elif m_idx == 3:
                expression = r**3*(75582*r**16 - 350064*r**14 + 680680*r**12 - 720720*r**10 + 450450*r**8 - 168168*r**6 + 36036*r**4 - 3960*r**2 + 165)
            elif m_idx == 5:
                expression = r**5*(50388*r**14 - 222768*r**12 + 408408*r**10 - 400400*r**8 + 225225*r**6 - 72072*r**4 + 12012*r**2 - 792)
            elif m_idx == 7:
                expression = r**7*(27132*r**12 - 111384*r**10 + 185640*r**8 - 160160*r**6 + 75075*r**4 - 18018*r**2 + 1716)
            elif m_idx == 9:
                expression = r**9*(11628*r**10 - 42840*r**8 + 61880*r**6 - 43680*r**4 + 15015*r**2 - 2002)
            elif m_idx == 11:
                expression = r**11*(3876*r**8 - 12240*r**6 + 14280*r**4 - 7280*r**2 + 1365)
            elif m_idx == 13:
                expression = r**13*(969*r**6 - 2448*r**4 + 2040*r**2 - 560)
            elif m_idx == 15:
                expression = r**15*(171*r**4 - 306*r**2 + 136)
            elif m_idx == 17:
                expression = r**17*(19*r**2 - 18)
            elif m_idx == 19:
                expression = r**19

        else:
            k = sympy.symbols('k', integer=True)
            # General case for radial polynomial with m_idx != 0
            expression = sympy.Sum((-1)**k * sympy.factorial(n_idx - k) / (sympy.factorial(k) * sympy.factorial((n_idx + abs(m_idx)) // 2 - k) * sympy.factorial((n_idx - abs(m_idx)) // 2 - k)) * r**(n_idx - 2 * k), (k, 0, (n_idx - abs(m_idx)) // 2))
            expression = sympy.simplify(expression.doit())

        # Derivative with respect to rho
        if rho_derivative_idx > 0:
            expression = sympy.diff(expression, r, rho_derivative_idx)

        # Theta part of the Zernike polynomial
        if not flag_radial:

            # Extract the angular theta_derivative
            theta_derivative_idx = theta_derivative[idx]

            # Construct the angular polynomial expression
            t = sympy.symbols('t')
            
            # According to the angular derivative, we compute the cosine factor
            if m_idx == 0:
                if theta_derivative_idx == 0:
                    cosine = sympy.sympify(1) + 0 * t
                else:
                    cosine = sympy.sympify(0) * t
                
            if m_idx > 0:
                if theta_derivative_idx == 0:
                    cosine = sympy.cos(abs(m_idx) * t)
                elif theta_derivative_idx % 4 == 0:
                    cosine = (abs(m_idx) ** theta_derivative_idx) * sympy.cos(abs(m_idx) * t)
                elif theta_derivative_idx % 4 == 1:
                    cosine = - (abs(m_idx) ** theta_derivative_idx) * sympy.sin(abs(m_idx) * t)
                elif theta_derivative_idx % 4 == 2:
                    cosine = - (abs(m_idx) ** theta_derivative_idx) * sympy.cos(abs(m_idx) * t)
                else:
                    cosine = (abs(m_idx) ** theta_derivative_idx) * sympy.sin(abs(m_idx) * t)
                
            if m_idx < 0:
                if theta_derivative_idx == 0:
                    cosine = sympy.sin(abs(m_idx) * t)
                elif theta_derivative_idx % 4 == 0:
                    cosine = (abs(m_idx) ** theta_derivative_idx) * sympy.sin(abs(m_idx) * t)
                elif theta_derivative_idx % 4 == 1:
                    cosine = (abs(m_idx) ** theta_derivative_idx) * sympy.cos(abs(m_idx) * t)
                elif theta_derivative_idx % 4 == 2:
                    cosine = - (abs(m_idx) ** theta_derivative_idx) * sympy.sin(abs(m_idx) * t)
                else:
                    cosine = - (abs(m_idx) ** theta_derivative_idx) * sympy.cos(abs(m_idx) * t)

            # Combine the radial and angular parts
            expression = cosine * expression

        # Compute the polynomial
        output.append(expression)

    return output


