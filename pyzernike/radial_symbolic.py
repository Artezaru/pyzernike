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

def radial_symbolic(
    n: Sequence[Integral],
    m: Sequence[Integral],
    rho_derivative: Optional[Sequence[Integral]] = None,
    _skip: bool = False
) -> List[sympy.Expr]:
    r"""
    Compute the symbolic expression of the radial Zernike polynomial :math:`R_{n}^{m}(\rho)` for :math:`\rho \leq 1`.

    The radial Zernike polynomial is defined as follows:

    .. math::

        R_{n}^{m}(\rho) = \sum_{k=0}^{(n-m)/2} \frac{(-1)^k (n-k)!}{k! ((n+m)/2 - k)! ((n-m)/2 - k)!} \rho^{n-2k}

    if :math:`n < 0`, :math:`m < 0`, :math:`n < m`, or :math:`(n - m)` is odd, the output is the zero polynomial.

    .. seealso::

        - :func:`pyzernike.zernike_symbolic` for computing the full Zernike polynomial symbolic expression :math:`Z_{n}^{m}(\rho, \theta)`.
        - :func:`pyzernike.core.core_symbolic` to inspect the core implementation of the symbolic computation.
        - The page :doc:`../../mathematical_description` in the documentation for the detailed mathematical description of the Zernike polynomials.
    
    The function allows to display several radial Zernike polynomials for different sets of (order, degree, derivative order) given as sequences.

    - The parameters ``n``, ``m`` and ``rho_derivative`` must be sequences of integers with the same length.

    The output is a list of sympy expressions, each containing the symbolic expression of the radial Zernike polynomial for the corresponding order and degree.
    The list has the same length as the input sequences.

    .. note::

        The symbol `r` is used to represent the radial coordinate :math:`\rho` in the symbolic expression.

    Parameters
    ----------
    n : Sequence[Integral]
        A list of the radial order(s) of the Zernike polynomial(s) to compute.

    m : Sequence[Integral]
        A list of the radial degree(s) of the Zernike polynomial(s) to compute.

    rho_derivative : Optional[Sequence[Integral]], optional
        A list of the order(s) of the radial derivative(s) to compute.
        If None, is it assumed that rho_derivative is 0 for all polynomials.

    Returns
    -------
    List[sympy.Expr]
        A list of symbolic expressions containing the radial Zernike polynomial values for each order and degree
        Each expression is a sympy expression that can be evaluated for specific values of :math:`\rho`.

    Raises
    ------
    TypeError
        If n, m or rho_derivative (if not None) are not sequences of integers.

    ValueError
        If the lengths of n, m and rho_derivative (if not None) are not the same.

    Examples
    --------
    Compute the expression of the radial Zernike polynomial :math:`R_{2}^{0}(\rho)`:

    .. code-block:: python

        from pyzernike import radial_symbolic
        result = radial_symbolic(n=[2], m=[0])
        expression = result[0]  # result is a list, we take the first element
        print(expression)

    .. code-block:: console

        2*r**2 - 1

    Then evaluate the expression for a specific value of :math:`\rho`:

    .. code-block:: python

        import numpy
        import sympy
        rho = numpy.linspace(0, 1, 100)
        # `r` represents the radial coordinate in the symbolic expression
        
        func = sympy.lambdify('r', expression, 'numpy')
        evaluated_result = func(rho)

    """
    if not _skip:
        if not isinstance(n, Sequence) or not all(isinstance(i, Integral) for i in n):
            raise TypeError("n must be a sequence of integers.")
        if not isinstance(m, Sequence) or not all(isinstance(i, Integral) for i in m):
            raise TypeError("m must be a sequence of integers.")
        if rho_derivative is not None:
            if not isinstance(rho_derivative, Sequence) or not all(isinstance(i, Integral) and i >= 0 for i in rho_derivative):
                raise TypeError("rho_derivative must be a sequence of non-negative integers.")
        
        if len(n) != len(m):
            raise ValueError("n and m must have the same length.")
        if rho_derivative is not None and len(n) != len(rho_derivative):
            raise ValueError("n and rho_derivative must have the same length.")
        if rho_derivative is None:
            rho_derivative = [0] * len(n)

    # Compute the radial polynomials using the core_polynomial function
    radial_expressions = core_symbolic(
        n=n,
        m=m,
        rho_derivative=rho_derivative,
        theta_derivative=None,
        flag_radial=True
    )

    # Return the radial polynomials
    return radial_expressions
