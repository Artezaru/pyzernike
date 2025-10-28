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
from typing import Sequence, Optional

from .core.core_display import core_display

def zernike_display(
    n: Sequence[Integral],
    m: Sequence[Integral],
    rho_derivative: Optional[Sequence[Integral]] = None,
    theta_derivative: Optional[Sequence[Integral]] = None,
    precompute: bool = True,
    _skip: bool = False
) -> None:
    r"""
    Display the Zernike polynomial :math:`Z_{n}^{m}(\rho, \theta)` for :math:`\rho \leq 1` and :math:`\theta \in [0, 2\pi]` in an interactive matplotlib figure.

    The Zernike polynomial is defined as follows:

    .. math::

        Z_{n}^{m}(\rho, \theta) = R_{n}^{m}(\rho) \cos(m \theta) \quad \text{if} \quad m \geq 0

    .. math::

        Z_{n}^{m}(\rho, \theta) = R_{n}^{-m}(\rho) \sin(-m \theta) \quad \text{if} \quad m < 0

    If :math:`n < 0`, :math:`n < |m|`, or :math:`(n - m)` is odd, the polynomial is zero.

    .. seealso::

        - :func:`pyzernike.radial_display` for displaying the radial part of the Zernike polynomial :math:`R_{n}^{m}(\rho)`.
        - :func:`pyzernike.core.core_display` to inspect the core implementation of the display.
        - The page :doc:`../../mathematical_description` in the documentation for the detailed mathematical description of the Zernike polynomials.

    The function allows to display several Zernike polynomials for different sets of (order, degree, derivative orders) given as sequences.

    - The parameters ``n``, ``m``, ``rho_derivative`` and ``theta_derivative`` must be sequences of integers with the same length.

    Parameters
    ----------
    n : Sequence[Integral]
        A list of the radial order(s) of the Zernike polynomial(s) to display.

    m : Sequence[Integral]
        A list of the radial degree(s) of the Zernike polynomial(s) to display.

    rho_derivative : Optional[Sequence[Integral]], optional
        A list of the order(s) of the radial derivative(s) to display.
        If None, is it assumed that rho_derivative is 0 for all polynomials.

    theta_derivative : Optional[Sequence[Integral]], optional
        A list of the order(s) of the angular derivative(s) to display.
        If None, is it assumed that theta_derivative is 0 for all polynomials.

    precompute : bool, optional
        If True, the useful terms for the Zernike polynomials are precomputed to optimize the computation.
        If False, the useful terms are computed on-the-fly to avoid memory overhead.
        Default is True.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If n, m, rho_derivative or theta_derivative (if not None) are not sequences of integers.

    ValueError
        If the lengths of n, m, rho_derivative and theta_derivative (if not None) are not the same.

    Examples
    --------
    Display the Zernike polynomial :math:`Z_{2}^{0}(\rho, \theta)`:

    .. code-block:: python

        from pyzernike import zernike_display
        zernike_display(n=[2], m=[0]) # This will display the Zernike polynomial Z_2^0 in an interactive matplotlib figure.

    To display multiple Zernike polynomials, you can pass sequences for `n` and `m`:

    .. code-block:: python

        from pyzernike import zernike_display
        zernike_display(n=[2, 3, 4], m=[0, 1, 2], rho_derivative=[0, 0, 1], theta_derivative=[0, 1, 0])

    .. image:: ../../../pyzernike/resources/zernike_display.png
        :width: 600px
        :align: center

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
        if not isinstance(precompute, bool):
            raise TypeError("precompute must be a boolean.")
        
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
    core_display(
        n=n,
        m=m,
        rho_derivative=rho_derivative,
        theta_derivative=theta_derivative,
        flag_radial=False,
        precompute=precompute,
    )