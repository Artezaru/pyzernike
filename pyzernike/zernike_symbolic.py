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

def zernike_symbolic(
    n: Union[numpy.array, Sequence[Integral]],
    m: Union[numpy.array, Sequence[Integral]],
    rho_derivative: Optional[Union[numpy.array, Sequence[Integral]]] = None,
    theta_derivative: Optional[Union[numpy.array, Sequence[Integral]]] = None,
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
    
    The function allows to display several Zernike polynomials for different sets of (order, azimuthal frequency, derivative orders) given as sequences.

    - The parameters ``n``, ``m``, ``rho_derivative`` and ``theta_derivative`` must be sequences of integers with the same length.

    The output is a list of sympy expressions, each containing the symbolic expression of the Zernike polynomial for the corresponding order and azimuthal frequency.
    The list has the same length as the input sequences.

    .. note::

        - The symbol `r` is used to represent the radial coordinate :math:`\rho` in the symbolic expression.
        - The symbol `t` is used to represent the angular coordinate :math:`\theta` in the symbolic expression.

    Parameters
    ----------
    n : Sequence[Integral] or numpy.array
        A sequence (List, Tuple) or 1D numpy array of the radial order(s) of the Zernike polynomial(s) to compute. Must be non-negative integers.

    m : Sequence[Integral] or numpy.array
        A sequence (List, Tuple) or 1D numpy array of the azimuthal frequency(ies) of the Zernike polynomial(s) to compute. Must be non-negative integers.

    rho_derivative : Optional[Union[Sequence[Integral], numpy.array]], optional
        A sequence (List, Tuple) or 1D numpy array of the order(s) of the radial derivative(s) to compute. Must be non-negative integers.
        If None, is it assumed that rho_derivative is 0 for all polynomials.

    theta_derivative : Optional[Union[Sequence[Integral], numpy.array]], optional
        A sequence (List, Tuple) or 1D numpy array of the order(s) of the angular derivative(s) to compute. Must be non-negative integers.
        If None, is it assumed that theta_derivative is 0 for all polynomials.

    Returns
    -------
    List[sympy.Expr]
        A list of symbolic expressions containing the Zernike polynomial values for each order and azimuthal frequency.
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
    # Check the input parameters
    if not isinstance(n, (Sequence, numpy.ndarray)) or not all(isinstance(i, Integral) for i in n):
        raise TypeError("n must be a sequence or a 1D array of integers.")
    if not isinstance(m, (Sequence, numpy.ndarray)) or not all(isinstance(i, Integral) for i in m):
        raise TypeError("m must be a sequence or a 1D array of integers.")
    if rho_derivative is not None:
        if not isinstance(rho_derivative, (Sequence, numpy.ndarray)) or not all(isinstance(i, Integral) and i >= 0 for i in rho_derivative):
            raise TypeError("rho_derivative must be a sequence or a 1D array of non-negative integers.")
    if theta_derivative is not None:
        if not isinstance(theta_derivative, (Sequence, numpy.ndarray)) or not all(isinstance(i, Integral) and i >= 0 for i in theta_derivative):
            raise TypeError("theta_derivative must be a sequence or a 1D array of non-negative integers.")
    
    # Convert n, m and rho_derivative to arrays for length checking
    n = numpy.asarray(n, dtype=numpy.int32)
    m = numpy.asarray(m, dtype=numpy.int32)
    if rho_derivative is not None:
        rho_derivative = numpy.asarray(rho_derivative, dtype=numpy.int32)
    else:
        rho_derivative = numpy.zeros_like(n, dtype=numpy.int32)
    if theta_derivative is not None:
        theta_derivative = numpy.asarray(theta_derivative, dtype=numpy.int32)
    else:
        theta_derivative = numpy.zeros_like(n, dtype=numpy.int32)

    # Check lengths
    if not n.ndim == 1:
        raise TypeError("n must be a sequence or a 1D array of integers.")
    if not m.ndim == 1:
        raise TypeError("m must be a sequence or a 1D array of integers.")
    if rho_derivative.ndim != 1:
        raise TypeError("rho_derivative must be a sequence or a 1D array of integers.")
    if theta_derivative.ndim != 1:
        raise TypeError("theta_derivative must be a sequence or a 1D array of integers.")

    if not (n.size == m.size == rho_derivative.size == theta_derivative.size):
        raise ValueError("n, m, rho_derivative (if given) and theta_derivative (if given) must have the same length.")

    # Compute the radial polynomials using the core_polynomial function
    radial_expressions = core_symbolic(
        n=n,
        m=m,
        rho_derivative=rho_derivative,
        theta_derivative=theta_derivative,
        flag_radial=False
    )

    # Return the radial polynomials
    return radial_expressions
