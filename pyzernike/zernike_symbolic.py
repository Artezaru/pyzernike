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

from numbers import Integral
from typing import Sequence, List, Optional
import sympy

from .core.core_symbolic import core_symbolic

def zernike_symbolic(
    n: Sequence[Integral],
    m: Sequence[Integral],
    rho_derivative: Optional[Sequence[Integral]] = None,
    theta_derivative: Optional[Sequence[Integral]] = None,
    _skip: bool = False
) -> List[sympy.Expr]:
    r"""
    Compute the symbolic expression of the Zernike polynomial :math:`Z_{n}^{m}(\rho, \theta)` for :math:`\rho \leq 1` and :math:`\theta \in [0, 2\pi]`.

    The Zernike polynomial is defined as follows:

    .. math::

        Z_{n}^{m}(\rho, \theta) = R_{n}^{m}(\rho) \cos(m \theta) \quad \text{if} \quad m \geq 0

    .. math::

        Z_{n}^{m}(\rho, \theta) = R_{n}^{-m}(\rho) \sin(-m \theta) \quad \text{if} \quad m < 0

    If :math:`n < 0`, :math:`n < |m|`, or :math:`(n - m)` is odd, the polynomial is zero.

    .. seealso::

        - :func:`pyzernike.radial_polynomial` for computing the radial part of the Zernike polynomial :math:`R_{n}^{m}(\rho)`.
        - :func:`pyzernike.core.core_polynomial` to inspect the core implementation of the computation.
        - The page :doc:`../../mathematical_description` in the documentation for the detailed mathematical description of the Zernike polynomials.
    
    The function allows to display several Zernike polynomials for different sets of (order, degree, derivative orders) given as sequences.

    - The parameters ``n``, ``m``, ``rho_derivative`` and ``theta_derivative`` must be sequences of integers with the same length.

    The output is a list of sympy expressions, each containing the symbolic expression of the Zernike polynomial for the corresponding order and degree.
    The list has the same length as the input sequences.

    .. note::

        - The symbol `r` is used to represent the radial coordinate :math:`\rho` in the symbolic expression.
        - The symbol `t` is used to represent the angular coordinate :math:`\theta` in the symbolic expression.

    Parameters
    ----------
    n : Sequence[Integral]
        A list of the radial order(s) of the Zernike polynomial(s) to compute.

    m : Sequence[Integral]
        A list of the radial degree(s) of the Zernike polynomial(s) to compute.

    rho_derivative : Optional[Sequence[Integral]], optional
        A list of the order(s) of the radial derivative(s) to compute.
        If None, is it assumed that rho_derivative is 0 for all polynomials.

    theta_derivative : Optional[Sequence[Integral]], optional
        A list of the order(s) of the angular derivative(s) to compute.
        If None, is it assumed that theta_derivative is 0 for all polynomials.

    Returns
    -------
    List[sympy.Expr]
        A list of symbolic expressions containing the Zernike polynomial values for each order and degree
        Each expression is a sympy expression that can be evaluated for specific values of :math:`\rho` and :math:`\theta`.

    Raises
    ------
    TypeError
        If n, m, rho_derivative or theta_derivative (if not None) are not sequences of integers.

    ValueError
        If the lengths of n, m, rho_derivative or theta_derivative (if not None) are not the same.

    Examples
    --------
    Compute the expression of the radial Zernike polynomial :math:`Z_{2}^{1}(\rho, \theta)`:

    .. code-block:: python

        from pyzernike import zernike_symbolic
        result = zernike_symbolic(n=[2], m=[1])
        expression = result[0]  # result is a list, we take the first element
        print(expression)

    .. code-block:: console

        (2*r**2 - 1)*cos(t)

    Then evaluate the expression for a specific value of :math:`\rho` and :math:`\theta`:

    .. code-block:: python

        import numpy
        import sympy
        rho = numpy.linspace(0, 1, 100)
        theta = numpy.linspace(0, 2 * numpy.pi, 100)

        # `r` represents the radial coordinate in the symbolic expression
        # `t` represents the angular coordinate in the symbolic expression

        func = sympy.lambdify(['r', 't'], expression, 'numpy')
        evaluated_result = func(rho, theta)

    """
    if not _skip:
        if not isinstance(n, Sequence) or not all(isinstance(i, Integral) for i in n):
            raise TypeError("n must be a sequence of integers.")
        if not isinstance(m, Sequence) or not all(isinstance(i, Integral) for i in m):
            raise TypeError("m must be a sequence of integers.")
        if rho_derivative is not None:
            if not isinstance(rho_derivative, Sequence) or not all(isinstance(i, Integral) and i >= 0 for i in rho_derivative):
                raise TypeError("rho_derivative must be a sequence of non-negative integers.")
        if theta_derivative is not None:
            if not isinstance(theta_derivative, Sequence) or not all(isinstance(i, Integral) and i >= 0 for i in theta_derivative):
                raise TypeError("theta_derivative must be a sequence of non-negative integers.")
        
        if len(n) != len(m):
            raise ValueError("n and m must have the same length.")
        if rho_derivative is not None and len(n) != len(rho_derivative):
            raise ValueError("n and rho_derivative must have the same length.")
        if theta_derivative is not None and len(n) != len(theta_derivative):
            raise ValueError("n and theta_derivative must have the same length.")
        if rho_derivative is None:
            rho_derivative = [0] * len(n)
        if theta_derivative is None:
            theta_derivative = [0] * len(n)

    # Compute the radial polynomials using the core_polynomial function
    zernike_expressions = core_symbolic(
        n=n,
        m=m,
        rho_derivative=rho_derivative,
        theta_derivative=theta_derivative,
        flag_radial=False
    )

    # Return the radial polynomials
    return zernike_expressions
