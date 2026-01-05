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
from typing import Sequence, List, Optional, Union
import numpy
import sympy

from .core.core_symbolic import core_symbolic

def radial_symbolic(
    n: Union[numpy.array, Sequence[Integral]],
    m: Union[numpy.array, Sequence[Integral]],
    rho_derivative: Optional[Union[numpy.array, Sequence[Integral]]] = None,
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
    
    The function allows to display several radial Zernike polynomials for different sets of (order, azimuthal frequency, derivative order) given as sequences.

    - The parameters ``n``, ``m`` and ``rho_derivative`` must be sequences of integers with the same length.

    The output is a list of sympy expressions, each containing the symbolic expression of the radial Zernike polynomial for the corresponding order and azimuthal frequency.
    The list has the same length as the input sequences.

    .. note::

        The symbol `r` is used to represent the radial coordinate :math:`\rho` in the symbolic expression.

    Parameters
    ----------
    n : Sequence[Integral] or numpy.array
        A sequence (List, Tuple) or 1D numpy array of the radial order(s) of the Zernike polynomial(s) to compute. Must be non-negative integers.

    m : Sequence[Integral] or numpy.array
        A sequence (List, Tuple) or 1D numpy array of the azimuthal frequency(ies) of the Zernike polynomial(s) to compute. Must be non-negative integers.

    rho_derivative : Optional[Union[Sequence[Integral], numpy.array]], optional
        A sequence (List, Tuple) or 1D numpy array of the order(s) of the radial derivative(s) to compute. Must be non-negative integers.
        If None, is it assumed that rho_derivative is 0 for all polynomials.

    Returns
    -------
    List[sympy.Expr]
        A list of symbolic expressions containing the radial Zernike polynomial values for each order and azimuthal frequency
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
    # Check the input parameters
    if not isinstance(n, (Sequence, numpy.ndarray)) or not all(isinstance(i, Integral) for i in n):
        raise TypeError("n must be a sequence or a 1D array of integers.")
    if not isinstance(m, (Sequence, numpy.ndarray)) or not all(isinstance(i, Integral) for i in m):
        raise TypeError("m must be a sequence or a 1D array of integers.")
    if rho_derivative is not None:
        if not isinstance(rho_derivative, (Sequence, numpy.ndarray)) or not all(isinstance(i, Integral) and i >= 0 for i in rho_derivative):
            raise TypeError("rho_derivative must be a sequence or a 1D array of non-negative integers.")
    
    # Convert n, m and rho_derivative to arrays for length checking
    n = numpy.asarray(n, dtype=numpy.int32)
    m = numpy.asarray(m, dtype=numpy.int32)
    if rho_derivative is not None:
        rho_derivative = numpy.asarray(rho_derivative, dtype=numpy.int32)
    else:
        rho_derivative = numpy.zeros_like(n, dtype=numpy.int32)

    # Check lengths
    if not n.ndim == 1:
        raise TypeError("n must be a sequence or a 1D array of integers.")
    if not m.ndim == 1:
        raise TypeError("m must be a sequence or a 1D array of integers.")
    if rho_derivative.ndim != 1:
        raise TypeError("rho_derivative must be a sequence or a 1D array of integers.")

    if not (n.size == m.size == rho_derivative.size):
        raise ValueError("n, m and rho_derivative (if given) must have the same length.")

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
