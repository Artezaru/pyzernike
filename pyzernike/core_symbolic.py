import numpy
from typing import Sequence, List
import sympy

def core_symbolic(
        n: Sequence[int], 
        m: Sequence[int], 
        rho_derivative: Sequence[int], 
        theta_derivative: Sequence[int],
        flag_radial: bool,
    ) -> List[sympy.Expr]:
    r"""

    .. warning::

        This method is a core function of ``pyzernike`` that computes Zernike polynomials for given `n`, `m`, `rho_derivative`, and `theta_derivative` values.
        It is not a method designed to be use by the users directly, but rather a helper function for the optimized radial polynomial computation.

        Please ensure that you understand the mathematical background of Zernike polynomials before using this function.
        ``n``, ``m``, ``rho_derivative`` and ``theta_derivative`` are expected to be sequences of integers with the same length and valid values.

    .. note::

        The `x` symbol is used to represent the radial coordinate :math:`\rho` in the symbolic expression.
        The `y` symbol is used to represent the angular coordinate :math:`\theta` in the symbolic expression.

    .. seealso::

        - :func:`pyzernike.radial_polynomial` for the radial Zernike polynomial computation.
        - :func:`pyzernike.zernike_polynomial` for the full Zernike polynomial computation.
        - :func:`pyzernike.xy_zernike_polynomial` for the Zernike polynomial computation in Cartesian coordinates.

    Compute the symbolic expression of the Zernike polynomial :math:`Z_{n}^{m}(\rho, \theta)` or the radial Zernike polynomial :math:`R_{n}^{m}(\rho)` if the flag `flag_radial` is set to True with symbolic ``sympy`` computation.

    The Zernike polynomial is defined as follows:

    .. math::

        Z_{n}^{m}(\rho, \theta) = R_{n}^{m}(\rho) \cos(m \theta) \quad \text{if} \quad m > 0
    
    .. math::

        Z_{n}^{m}(\rho, \theta) = R_{n}^{-m}(\rho) \sin(-m \theta) \quad \text{if} \quad m < 0

    The derivative of order (derivative (a)) of the Zernike polynomial with respect to rho and order (derivative (b)) with respect to theta is defined as follows :

    .. math::

        \frac{\partial^{a}\partial^{b}Z_{n}^{m}(\rho, \theta)}{\partial \rho^{a} \partial \theta^{b}} = \frac{\partial^{a}R_{n}^{m}(\rho)}{\partial \rho^{a}} \frac{\partial^{b}\cos(m \theta)}{\partial \theta^{b}} \quad \text{if} \quad m > 0

    If :math:`|m| > n` or :math:`n < 0`, or :math:`(n - m)` is odd, the output is a zeros array with the given shape.

    The radial Zernike polynomial is defined as follows:

    .. math::

        R_{n}^{m}(\rho) = \sum_{k=0}^{(n-m)/2} \frac{(-1)^k (n-k)!}{k! ((n+m)/2 - k)! ((n-m)/2 - k)!} \rho^{n-2k}

    The derivative of order (derivative (a)) of the radial Zernike polynomial is defined as follows :

    .. math::

        \frac{d^{a}R_{n}^{m}(\rho)}{d\rho^{a}} = \sum_{k=0}^{(n-m)/2} \frac{(-1)^k (n-k)!}{k! ((n+m)/2 - k)! ((n-m)/2 - k)!} (n-2k) (n-2k-1) \ldots (n-2k-a+1) \rho^{n-2k-a}

    Parameters
    ----------    
    n : Sequence[int]
        A list of integers containing the order `n` of each radial Zernike polynomial to compute.

    m : Sequence[int]
        A list of integers containing the degree `m` of each radial Zernike polynomial to compute.

    rho_derivative : Sequence[int]
        A list of integers containing the order of the radial derivative to compute for each radial Zernike polynomial.
        If `rho_derivative` is None, no radial derivative is computed.

    theta_derivative : Sequence[int]
        A list of integers containing the order of the angular derivative to compute for each Zernike polynomial.
        If `theta_derivative` is None, no angular derivative is computed. ONLY USED IF `flag_radial` IS False.

    flag_radial : bool
        If True, the radial Zernike polynomial :math:`R_{n}^{m}(\rho)` is computed instead of the full Zernike polynomial :math:`Z_{n}^{m}(\rho, \theta)`.
        If False, the full Zernike polynomial :math:`Z_{n}^{m}(\rho, \theta)` is computed, which includes the angular part with the cosine and sine terms.

    Returns
    -------
    List[sympy.Expr]
        A list of symbolic expressions of the Zernike polynomial or radial Zernike polynomial for each order and degree specified in `n` and `m`.
        Each expression is a sympy expression that can be evaluated or manipulated further.
    """
    # Create the output list
    output = []

    # =================================================================
    # Boucle over the polynomials to compute
    # =================================================================
    for idx in range(len(n)):
        n_idx = n[idx]
        m_idx = m[idx]
        rho_derivative_idx = rho_derivative[idx]

        # Construct the radial polynomial expression
        x = sympy.symbols('x')

        if n_idx < 0 or (n_idx - m_idx) % 2 != 0 or abs(m_idx) > n_idx:
            expression = sympy.sympify(0) * x # Need to lambdify to work correctly (must contain x)

        # Case for derivatives of order greater than n_idx
        elif rho_derivative_idx > n_idx:
            expression = sympy.sympify(0) * x

        # Case for radial polynomial only
        elif flag_radial and m_idx < 0:
            expression = sympy.sympify(0) * x

        # Compute the symbolic expression for the radial polynomial with derivative = 0
        elif n_idx == 0:
            if abs(m_idx) == 0:
                expression = sympy.sympify(1) + 0 * x

        elif n_idx == 1:
            if m_idx == 1:
                expression = x

        elif n_idx == 2:
            if m_idx == 0:
                expression = 2*x**2 - 1
            elif m_idx == 2:
                expression = x**2

        elif n_idx == 3:
            if m_idx == 1:
                expression = 3*x**3 - 2*x
            elif m_idx == 3:
                expression = x**3

        elif n_idx == 4:
            if m_idx == 0:
                expression = 6*x**4 - 6*x**2 + 1
            elif m_idx == 2:
                expression = 4*x**4 - 3*x**2
            elif m_idx == 4:
                expression = x**4

        elif n_idx == 5:
            if m_idx == 1:
                expression = 10*x**5 - 12*x**3 + 3*x
            elif m_idx == 3:
                expression = 5*x**5 - 4*x**3
            elif m_idx == 5:
                expression = x**5

        elif n_idx == 6:
            if m_idx == 0:
                expression = 20*x**6 - 30*x**4 + 12*x**2 - 1
            elif m_idx == 2:
                expression = 15*x**6 - 20*x**4 + 6*x**2
            elif m_idx == 4:
                expression = 6*x**6 - 5*x**4
            elif m_idx == 6:
                expression = x**6

        elif n_idx == 7:
            if m_idx == 1:
                expression = 35*x**7 - 60*x**5 + 30*x**3 - 4*x
            elif m_idx == 3:
                expression = 21*x**7 - 30*x**5 + 10*x**3
            elif m_idx == 5:
                expression = 7*x**7 - 6*x**5
            elif m_idx == 7:
                expression = x**7

        elif n_idx == 8:
            if m_idx == 0:
                expression = 70*x**8 - 140*x**6 + 90*x**4 - 20*x**2 + 1
            elif m_idx == 2:
                expression = 56*x**8 - 105*x**6 + 60*x**4 - 10*x**2
            elif m_idx == 4:
                expression = 28*x**8 - 42*x**6 + 15*x**4
            elif m_idx == 6:
                expression = 8*x**8 - 7*x**6
            elif m_idx == 8:
                expression = x**8

        elif n_idx == 9:
            if m_idx == 1:
                expression = 126*x**9 - 280*x**7 + 210*x**5 - 60*x**3 + 5*x
            elif m_idx == 3:
                expression = 84*x**9 - 168*x**7 + 105*x**5 - 20*x**3
            elif m_idx == 5:
                expression = 36*x**9 - 56*x**7 + 21*x**5
            elif m_idx == 7:
                expression = 9*x**9 - 8*x**7
            elif m_idx == 9:
                expression = x**9

        elif n_idx == 10:
            if m_idx == 0:
                expression = 252*x**10 - 630*x**8 + 560*x**6 - 210*x**4 + 30*x**2 - 1
            elif m_idx == 2:
                expression = x**2*(210*x**8 - 504*x**6 + 420*x**4 - 140*x**2 + 15)
            elif m_idx == 4:
                expression = x**4*(120*x**6 - 252*x**4 + 168*x**2 - 35)
            elif m_idx == 6:
                expression = x**6*(45*x**4 - 72*x**2 + 28)
            elif m_idx == 8:
                expression = x**8*(10*x**2 - 9)
            elif m_idx == 10:
                expression = x**10

        elif n_idx == 11:
            if m_idx == 1:
                expression = x*(462*x**10 - 1260*x**8 + 1260*x**6 - 560*x**4 + 105*x**2 - 6)
            elif m_idx == 3:
                expression = x**3*(330*x**8 - 840*x**6 + 756*x**4 - 280*x**2 + 35)
            elif m_idx == 5:
                expression = x**5*(165*x**6 - 360*x**4 + 252*x**2 - 56)
            elif m_idx == 7:
                expression = x**7*(55*x**4 - 90*x**2 + 36)
            elif m_idx == 9:
                expression = x**9*(11*x**2 - 10)
            elif m_idx == 11:
                expression = x**11

        elif n_idx == 12:
            if m_idx == 0:
                expression = 924*x**12 - 2772*x**10 + 3150*x**8 - 1680*x**6 + 420*x**4 - 42*x**2 + 1
            elif m_idx == 2:
                expression = x**2*(792*x**10 - 2310*x**8 + 2520*x**6 - 1260*x**4 + 280*x**2 - 21)
            elif m_idx == 4:
                expression = x**4*(495*x**8 - 1320*x**6 + 1260*x**4 - 504*x**2 + 70)
            elif m_idx == 6:
                expression = x**6*(220*x**6 - 495*x**4 + 360*x**2 - 84)
            elif m_idx == 8:
                expression = x**8*(66*x**4 - 110*x**2 + 45)
            elif m_idx == 10:
                expression = x**10*(12*x**2 - 11)
            elif m_idx == 12:
                expression = x**12

        elif n_idx == 13:
            if m_idx == 1:
                expression = x*(1716*x**12 - 5544*x**10 + 6930*x**8 - 4200*x**6 + 1260*x**4 - 168*x**2 + 7)
            elif m_idx == 3:
                expression = x**3*(1287*x**10 - 3960*x**8 + 4620*x**6 - 2520*x**4 + 630*x**2 - 56)
            elif m_idx == 5:
                expression = x**5*(715*x**8 - 1980*x**6 + 1980*x**4 - 840*x**2 + 126)
            elif m_idx == 7:
                expression = x**7*(286*x**6 - 660*x**4 + 495*x**2 - 120)
            elif m_idx == 9:
                expression = x**9*(78*x**4 - 132*x**2 + 55)
            elif m_idx == 11:
                expression = x**11*(13*x**2 - 12)
            elif m_idx == 13:
                expression = x**13

        elif n_idx == 14:
            if m_idx == 0:
                expression = 3432*x**14 - 12012*x**12 + 16632*x**10 - 11550*x**8 + 4200*x**6 - 756*x**4 + 56*x**2 - 1
            elif m_idx == 2:
                expression = x**2*(3003*x**12 - 10296*x**10 + 13860*x**8 - 9240*x**6 + 3150*x**4 - 504*x**2 + 28)
            elif m_idx == 4:
                expression = x**4*(2002*x**10 - 6435*x**8 + 7920*x**6 - 4620*x**4 + 1260*x**2 - 126)
            elif m_idx == 6:
                expression = x**6*(1001*x**8 - 2860*x**6 + 2970*x**4 - 1320*x**2 + 210)
            elif m_idx == 8:
                expression = x**8*(364*x**6 - 858*x**4 + 660*x**2 - 165)
            elif m_idx == 10:
                expression = x**10*(91*x**4 - 156*x**2 + 66)
            elif m_idx == 12:
                expression = x**12*(14*x**2 - 13)
            elif m_idx == 14:
                expression = x**14

        elif n_idx == 15:
            if m_idx == 1:
                expression = x*(6435*x**14 - 24024*x**12 + 36036*x**10 - 27720*x**8 + 11550*x**6 - 2520*x**4 + 252*x**2 - 8)
            elif m_idx == 3:
                expression = x**3*(5005*x**12 - 18018*x**10 + 25740*x**8 - 18480*x**6 + 6930*x**4 - 1260*x**2 + 84)
            elif m_idx == 5:
                expression = x**5*(3003*x**10 - 10010*x**8 + 12870*x**6 - 7920*x**4 + 2310*x**2 - 252)
            elif m_idx == 7:
                expression = x**7*(1365*x**8 - 4004*x**6 + 4290*x**4 - 1980*x**2 + 330)
            elif m_idx == 9:
                expression = x**9*(455*x**6 - 1092*x**4 + 858*x**2 - 220)
            elif m_idx == 11:
                expression = x**11*(105*x**4 - 182*x**2 + 78)
            elif m_idx == 13:
                expression = x**13*(15*x**2 - 14)
            elif m_idx == 15:
                expression = x**15

        elif n_idx == 16:
            if m_idx == 0:
                expression = 12870*x**16 - 51480*x**14 + 84084*x**12 - 72072*x**10 + 34650*x**8 - 9240*x**6 + 1260*x**4 - 72*x**2 + 1
            elif m_idx == 2:
                expression = x**2*(11440*x**14 - 45045*x**12 + 72072*x**10 - 60060*x**8 + 27720*x**6 - 6930*x**4 + 840*x**2 - 36)
            elif m_idx == 4:
                expression = x**4*(8008*x**12 - 30030*x**10 + 45045*x**8 - 34320*x**6 + 13860*x**4 - 2772*x**2 + 210)
            elif m_idx == 6:
                expression = x**6*(4368*x**10 - 15015*x**8 + 20020*x**6 - 12870*x**4 + 3960*x**2 - 462)
            elif m_idx == 8:
                expression = x**8*(1820*x**8 - 5460*x**6 + 6006*x**4 - 2860*x**2 + 495)
            elif m_idx == 10:
                expression = x**10*(560*x**6 - 1365*x**4 + 1092*x**2 - 286)
            elif m_idx == 12:
                expression = x**12*(120*x**4 - 210*x**2 + 91)
            elif m_idx == 14:
                expression = x**14*(16*x**2 - 15)
            elif m_idx == 16:
                expression = x**16

        elif n_idx == 17:
            if m_idx == 1:
                expression = x*(24310*x**16 - 102960*x**14 + 180180*x**12 - 168168*x**10 + 90090*x**8 - 27720*x**6 + 4620*x**4 - 360*x**2 + 9)
            elif m_idx == 3:
                expression = x**3*(19448*x**14 - 80080*x**12 + 135135*x**10 - 120120*x**8 + 60060*x**6 - 16632*x**4 + 2310*x**2 - 120)
            elif m_idx == 5:
                expression = x**5*(12376*x**12 - 48048*x**10 + 75075*x**8 - 60060*x**6 + 25740*x**4 - 5544*x**2 + 462)
            elif m_idx == 7:
                expression = x**7*(6188*x**10 - 21840*x**8 + 30030*x**6 - 20020*x**4 + 6435*x**2 - 792)
            elif m_idx == 9:
                expression = x**9*(2380*x**8 - 7280*x**6 + 8190*x**4 - 4004*x**2 + 715)
            elif m_idx == 11:
                expression = x**11*(680*x**6 - 1680*x**4 + 1365*x**2 - 364)
            elif m_idx == 13:
                expression = x**13*(136*x**4 - 240*x**2 + 105)
            elif m_idx == 15:
                expression = x**15*(17*x**2 - 16)
            elif m_idx == 17:
                expression = x**17

        elif n_idx == 18:
            if m_idx == 0:
                expression = 48620*x**18 - 218790*x**16 + 411840*x**14 - 420420*x**12 + 252252*x**10 - 90090*x**8 + 18480*x**6 - 1980*x**4 + 90*x**2 - 1
            elif m_idx == 2:
                expression = x**2*(43758*x**16 - 194480*x**14 + 360360*x**12 - 360360*x**10 + 210210*x**8 - 72072*x**6 + 13860*x**4 - 1320*x**2 + 45)
            elif m_idx == 4:
                expression = x**4*(31824*x**14 - 136136*x**12 + 240240*x**10 - 225225*x**8 + 120120*x**6 - 36036*x**4 + 5544*x**2 - 330)
            elif m_idx == 6:
                expression = x**6*(18564*x**12 - 74256*x**10 + 120120*x**8 - 100100*x**6 + 45045*x**4 - 10296*x**2 + 924)
            elif m_idx == 8:
                expression = x**8*(8568*x**10 - 30940*x**8 + 43680*x**6 - 30030*x**4 + 10010*x**2 - 1287)
            elif m_idx == 10:
                expression = x**10*(3060*x**8 - 9520*x**6 + 10920*x**4 - 5460*x**2 + 1001)
            elif m_idx == 12:
                expression = x**12*(816*x**6 - 2040*x**4 + 1680*x**2 - 455)
            elif m_idx == 14:
                expression = x**14*(153*x**4 - 272*x**2 + 120)
            elif m_idx == 16:
                expression = x**16*(18*x**2 - 17)
            elif m_idx == 18:
                expression = x**18

        elif n_idx == 19:
            if m_idx == 1:
                expression = x*(92378*x**18 - 437580*x**16 + 875160*x**14 - 960960*x**12 + 630630*x**10 - 252252*x**8 + 60060*x**6 - 7920*x**4 + 495*x**2 - 10)
            elif m_idx == 3:
                expression = x**3*(75582*x**16 - 350064*x**14 + 680680*x**12 - 720720*x**10 + 450450*x**8 - 168168*x**6 + 36036*x**4 - 3960*x**2 + 165)
            elif m_idx == 5:
                expression = x**5*(50388*x**14 - 222768*x**12 + 408408*x**10 - 400400*x**8 + 225225*x**6 - 72072*x**4 + 12012*x**2 - 792)
            elif m_idx == 7:
                expression = x**7*(27132*x**12 - 111384*x**10 + 185640*x**8 - 160160*x**6 + 75075*x**4 - 18018*x**2 + 1716)
            elif m_idx == 9:
                expression = x**9*(11628*x**10 - 42840*x**8 + 61880*x**6 - 43680*x**4 + 15015*x**2 - 2002)
            elif m_idx == 11:
                expression = x**11*(3876*x**8 - 12240*x**6 + 14280*x**4 - 7280*x**2 + 1365)
            elif m_idx == 13:
                expression = x**13*(969*x**6 - 2448*x**4 + 2040*x**2 - 560)
            elif m_idx == 15:
                expression = x**15*(171*x**4 - 306*x**2 + 136)
            elif m_idx == 17:
                expression = x**17*(19*x**2 - 18)
            elif m_idx == 19:
                expression = x**19

        else:
            k = sympy.symbols('k', integer=True)
            # General case for radial polynomial with m_idx != 0
            expression = sympy.Sum((-1)**k * sympy.factorial(n_idx - k) / (sympy.factorial(k) * sympy.factorial((n_idx + abs(m_idx)) // 2 - k) * sympy.factorial((n_idx - abs(m_idx)) // 2 - k)) * x**(n_idx - 2 * k), (k, 0, (n_idx - abs(m_idx)) // 2))
            expression = sympy.simplify(expression.doit())

        # Derivative with respect to rho
        if rho_derivative_idx > 0:
            expression = sympy.diff(expression, x, rho_derivative_idx)

        # Theta part of the Zernike polynomial
        if not flag_radial:
            theta_derivative_idx = theta_derivative[idx]

            # Construct the angular polynomial expression
            y = sympy.symbols('y')
            
            # According to the angular derivative, we compute the cosine factor
            if m_idx == 0:
                if theta_derivative_idx == 0:
                    cosine = sympy.sympify(1) + 0 * y
                else:
                    cosine = sympy.sympify(0) * y
                
            if m_idx > 0:
                if theta_derivative_idx == 0:
                    cosine = sympy.cos(abs(m_idx) * y)
                elif theta_derivative_idx % 4 == 0:
                    cosine = (abs(m_idx) ** theta_derivative_idx) * sympy.cos(abs(m_idx) * y)
                elif theta_derivative_idx % 4 == 1:
                    cosine = - (abs(m_idx) ** theta_derivative_idx) * sympy.sin(abs(m_idx) * y)
                elif theta_derivative_idx % 4 == 2:
                    cosine = - (abs(m_idx) ** theta_derivative_idx) * sympy.cos(abs(m_idx) * y)
                else:
                    cosine = (abs(m_idx) ** theta_derivative_idx) * sympy.sin(abs(m_idx) * y)
                
            if m_idx < 0:
                if theta_derivative_idx == 0:
                    cosine = sympy.sin(abs(m_idx) * y)
                elif theta_derivative_idx % 4 == 0:
                    cosine = (abs(m_idx) ** theta_derivative_idx) * sympy.sin(abs(m_idx) * y)
                elif theta_derivative_idx % 4 == 1:
                    cosine = (abs(m_idx) ** theta_derivative_idx) * sympy.cos(abs(m_idx) * y)
                elif theta_derivative_idx % 4 == 2:
                    cosine = - (abs(m_idx) ** theta_derivative_idx) * sympy.sin(abs(m_idx) * y)
                else:
                    cosine = - (abs(m_idx) ** theta_derivative_idx) * sympy.cos(abs(m_idx) * y)

            # Combine the radial and angular parts
            expression = cosine * expression

        # Compute the polynomial
        output.append(expression)

    return output


