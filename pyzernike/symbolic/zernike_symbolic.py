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

from ..core.parsing import _broadcast_indices
from .core_symbolic import _core_symbolic


def zernike_symbolic(
    n: Union[Integral, Sequence[Integral], numpy.ndarray],
    m: Union[Integral, Sequence[Integral], numpy.ndarray],
    rho_derivative: Optional[Union[Integral, Sequence[Integral], numpy.ndarray]] = 0,
    theta_derivative: Optional[Union[Integral, Sequence[Integral], numpy.ndarray]] = 0,
) -> Union[sympy.Expr, List[sympy.Expr]]:
    r"""
    Compute the symbolic expressions of the Zernike polynomials :math:`Z_{n}^{m}(\rho, \theta)` and of their derivatives.

    The Zernike polynomials [1]_ are defined on the unit disk for :math:`|m| \leq n` and :math:`n - m` even by:

    .. math::

        Z_{n}^{m}(\rho, \theta) = \begin{cases}
            R_{n}^{m}(\rho) \cos(m \theta) & \text{if } m \geq 0 \\
            R_{n}^{|m|}(\rho) \sin(|m| \theta) & \text{if } m < 0
        \end{cases}

    where :math:`R_{n}^{m}` is the radial polynomial (see :func:`pyzernike.radial_symbolic`). The
    polynomials are not normalized: multiply an expression by the factor of
    :func:`pyzernike.zernike_normalization` to get the normalized polynomial.

    The derivative of order :math:`a` with respect to :math:`\rho` and :math:`b` with respect to :math:`\theta` is:

    .. math::

        \frac{\partial^{a+b} Z_{n}^{m}}{\partial \rho^{a} \partial \theta^{b}} = \begin{cases}
            \frac{d^a R_{n}^{m}}{d\rho^a}(\rho) \, m^{b} \cos\left(m\theta + b\frac{\pi}{2}\right) & \text{if } m > 0 \\
            \frac{d^a R_{n}^{0}}{d\rho^a}(\rho) \, \delta_{b,0} & \text{if } m = 0 \\
            \frac{d^a R_{n}^{|m|}}{d\rho^a}(\rho) \, |m|^{b} \sin\left(|m|\theta + b\frac{\pi}{2}\right) & \text{if } m < 0
        \end{cases}

    where the phase shift :math:`b\frac{\pi}{2}` is written exactly as a sign and a swap between
    ``cos`` and ``sin``. The expressions use the symbols ``r`` for :math:`\rho` and ``t`` for :math:`\theta`.

    - If :math:`|m| > n` or :math:`n - m` is odd, the expression is ``0``.
    - If :math:`a > n`, or :math:`m = 0` and :math:`b > 0`, the expression is ``0``.

    **Shape of the inputs and of the output.** ``n``, ``m``, ``rho_derivative`` and ``theta_derivative``
    can each be a single integer or a sequence of integers (list, tuple or 1D numpy array):

    - If all of them are integers, a single expression is returned.
    - If at least one of them is a sequence, the integers are repeated to the length :math:`N` of the
      sequences (which must all have the same length), and a list of :math:`N` expressions is returned.

    .. seealso::

        - :func:`pyzernike.zernike_polynomial` for the numerical evaluation of the Zernike polynomials.
        - :func:`pyzernike.radial_symbolic` for the symbolic expressions of the radial polynomials.
        - :func:`pyzernike.zernike_normalization` for the normalization factors of Noll.
        - The page :doc:`../../mathematical_description/conventions` for the definition, the indexing and the
          normalization of the Zernike polynomials.
        - The page :doc:`../../mathematical_description/polynomials` for the derivatives of the Zernike polynomials.

    Parameters
    ----------
    n : Union[Integral, Sequence[Integral], numpy.ndarray]
        The radial order(s) :math:`n`.

    m : Union[Integral, Sequence[Integral], numpy.ndarray]
        The azimuthal frequency(ies) :math:`m`.

    rho_derivative : Optional[Union[Integral, Sequence[Integral], numpy.ndarray]], optional
        The order(s) :math:`a \geq 0` of the derivative with respect to :math:`\rho`. Default is 0 (no derivative).
        None is accepted and means 0.

    theta_derivative : Optional[Union[Integral, Sequence[Integral], numpy.ndarray]], optional
        The order(s) :math:`b \geq 0` of the derivative with respect to :math:`\theta`. Default is 0 (no derivative).
        None is accepted and means 0.

    Returns
    -------
    Union[sympy.Expr, List[sympy.Expr]]
        - If ``n``, ``m``, ``rho_derivative`` and ``theta_derivative`` are integers: a single expression in the symbols ``r`` and ``t``.
        - Otherwise: a list of :math:`N` expressions, where ``output[i]`` corresponds to
          ``(n[i], m[i], rho_derivative[i], theta_derivative[i])``.

    Raises
    ------
    TypeError
        If ``n``, ``m``, ``rho_derivative`` or ``theta_derivative`` is not an integer or a sequence of integers.

    ValueError
        If the sequences do not have the same length, or if a derivative order is negative.

    References
    ----------
    .. [1] F. Zernike, "Beugungstheorie des Schneidenverfahrens und seiner verbesserten Form, der
       Phasenkontrastmethode," Physica 1(7-12), 689-704 (1934). https://doi.org/10.1016/S0031-8914(34)80259-5

    Examples
    --------
    Get the expression of the coma :math:`Z_{3}^{1}(\rho, \theta)` and of its derivative with respect to :math:`\theta`:

    .. code-block:: python

        from pyzernike import zernike_symbolic

        print(zernike_symbolic(n=3, m=1))
        print(zernike_symbolic(n=3, m=1, theta_derivative=1))

    .. code-block:: console

        (3*r**3 - 2*r)*cos(t)
        -(3*r**3 - 2*r)*sin(t)

    Evaluate the gradient in polar coordinates with ``numpy``:

    .. code-block:: python

        import numpy
        import sympy

        expressions = zernike_symbolic(n=3, m=1, rho_derivative=[1, 0], theta_derivative=[0, 1])
        rho, theta = numpy.meshgrid(numpy.linspace(0, 1, 100), numpy.linspace(0, 2 * numpy.pi, 100))
        dZ_drho, dZ_dtheta = [sympy.lambdify(['r', 't'], expression, 'numpy')(rho, theta) for expression in expressions]

    .. note::

        For an expression equal to a constant (e.g. ``0``), ``sympy.lambdify`` returns a scalar and
        not an array. Use ``numpy.broadcast_to(value, rho.shape)`` if an array is needed.

    """
    indices, single = _broadcast_indices(
        integers={"n": n, "m": m},
        orders={"rho_derivative": rho_derivative, "theta_derivative": theta_derivative},
    )

    expressions = _core_symbolic(
        n=indices["n"],
        m=indices["m"],
        rho_derivative=indices["rho_derivative"],
        theta_derivative=indices["theta_derivative"],
        flag_radial=False,
    )

    return expressions[0] if single else expressions