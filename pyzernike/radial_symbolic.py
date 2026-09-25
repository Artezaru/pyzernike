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
from typing import List, Optional, Sequence, Union

import numpy
import sympy

from .core.parsing import _broadcast_integers
from .core.core_symbolic import _core_symbolic


def radial_symbolic(
    n: Union[Integral, Sequence[Integral], numpy.ndarray],
    m: Union[Integral, Sequence[Integral], numpy.ndarray],
    rho_derivative: Optional[Union[Integral, Sequence[Integral], numpy.ndarray]] = 0,
) -> Union[sympy.Expr, List[sympy.Expr]]:
    r"""
    Compute the symbolic expressions of the radial Zernike polynomials :math:`R_{n}^{m}(\rho)` and of their derivatives.

    The radial Zernike polynomials [1]_ are defined for :math:`0 \leq m \leq n` and :math:`n - m` even by:

    .. math::

        R_{n}^{m}(\rho) = \sum_{k=0}^{\frac{n-m}{2}} \frac{(-1)^k (n-k)!}{k! \left(\frac{n+m}{2} - k\right)! \left(\frac{n-m}{2} - k\right)!} \rho^{n-2k}

    The expressions are exact polynomials with integer coefficients in the symbol ``r``, which stands
    for :math:`\rho`. For :math:`n \leq 19` without derivative, they are read from a precomputed table
    (factorized form :math:`r^{m} P(r^2)` for :math:`n \geq 10`). Otherwise they are built in expanded
    form from the exact coefficients (see :func:`pyzernike.core._core_symbolic`).

    - If :math:`m < 0`, :math:`m > n` or :math:`n - m` is odd, the expression is ``0``.
    - If the derivative order :math:`a` is greater than :math:`n`, the expression is ``0``.

    **Shape of the inputs and of the output.** ``n``, ``m`` and ``rho_derivative`` can each be a single
    integer or a sequence of integers (list, tuple or 1D numpy array):

    - If all of them are integers, a single expression is returned.
    - If at least one of them is a sequence, the integers are repeated to the length :math:`N` of the
      sequences (which must all have the same length), and a list of :math:`N` expressions is returned.

    .. seealso::

        - :func:`pyzernike.radial_polynomial` for the numerical evaluation of the radial polynomials.
        - :func:`pyzernike.zernike_symbolic` for the symbolic expressions of the full Zernike polynomials.
        - The page :doc:`../../mathematical_description` for the mathematical description of the Zernike polynomials.

    Parameters
    ----------
    n : Union[Integral, Sequence[Integral], numpy.ndarray]
        The radial order(s) :math:`n`.

    m : Union[Integral, Sequence[Integral], numpy.ndarray]
        The azimuthal frequency(ies) :math:`m`.

    rho_derivative : Optional[Union[Integral, Sequence[Integral], numpy.ndarray]], optional
        The order(s) :math:`a \geq 0` of the derivative with respect to :math:`\rho`. Default is 0 (no derivative).
        None is accepted and means 0.

    Returns
    -------
    Union[sympy.Expr, List[sympy.Expr]]
        - If ``n``, ``m`` and ``rho_derivative`` are integers: a single expression in the symbol ``r``.
        - Otherwise: a list of :math:`N` expressions, where ``output[i]`` corresponds to ``(n[i], m[i], rho_derivative[i])``.

    Raises
    ------
    TypeError
        If ``n``, ``m`` or ``rho_derivative`` is not an integer or a sequence of integers.

    ValueError
        If the sequences do not have the same length, or if a derivative order is negative.

    References
    ----------
    .. [1] F. Zernike, "Beugungstheorie des Schneidenverfahrens und seiner verbesserten Form, der
       Phasenkontrastmethode," Physica 1(7-12), 689-704 (1934). https://doi.org/10.1016/S0031-8914(34)80259-5

    Examples
    --------
    Get the expression of :math:`R_{4}^{0}(\rho)` and of its first derivative:

    .. code-block:: python

        from pyzernike import radial_symbolic

        print(radial_symbolic(n=4, m=0))
        print(radial_symbolic(n=4, m=0, rho_derivative=1))

    .. code-block:: console

        6*r**4 - 6*r**2 + 1
        24*r**3 - 12*r

    Evaluate the expressions of order 4 with ``numpy``:

    .. code-block:: python

        import numpy
        import sympy

        expressions = radial_symbolic(n=4, m=[0, 2, 4])
        rho = numpy.linspace(0, 1, 100)
        values = [sympy.lambdify('r', expression, 'numpy')(rho) for expression in expressions]

    .. note::

        For an expression equal to a constant (e.g. ``0``), ``sympy.lambdify`` returns a scalar and
        not an array. Use ``numpy.broadcast_to(value, rho.shape)`` if an array is needed.

    """
    if rho_derivative is None:
        rho_derivative = 0

    indices, single = _broadcast_integers(n=n, m=m, rho_derivative=rho_derivative)
    if any(a < 0 for a in indices["rho_derivative"]):
        raise ValueError("rho_derivative must contain non-negative integers.")

    expressions = _core_symbolic(indices["n"], indices["m"], indices["rho_derivative"], None, True)

    return expressions[0] if single else expressions