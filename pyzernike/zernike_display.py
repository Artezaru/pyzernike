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
from typing import Sequence, Optional, Union
import numpy

from .core.core_display import core_display
from .core.core_corresponding_signed_integer_type import core_corresponding_signed_integer_type

def zernike_display(
    n: Union[numpy.array, Sequence[Integral]],
    m: Union[numpy.array, Sequence[Integral]],
    rho_derivative: Optional[Union[numpy.array, Sequence[Integral]]] = None,
    theta_derivative: Optional[Union[numpy.array, Sequence[Integral]]] = None,
    precompute: bool = True,
    float_type: numpy.floating = numpy.float64
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

    The function allows to display several Zernike polynomials for different sets of (order, azimuthal frequency, derivative orders) given as sequences.

    - The parameters ``n``, ``m``, ``rho_derivative`` and ``theta_derivative`` must be sequences of integers with the same length.

    Parameters
    ----------
    n : Sequence[Integral] or numpy.array
        A sequence (List, Tuple) or 1D numpy array of the radial order(s) of the Zernike polynomial(s) to display. Must be non-negative integers.

    m : Sequence[Integral] or numpy.array
        A sequence (List, Tuple) or 1D numpy array of the azimuthal frequency(ies) of the Zernike polynomial(s) to display. Must be non-negative integers.

    rho_derivative : Optional[Union[Sequence[Integral], numpy.array]], optional
        A sequence (List, Tuple) or 1D numpy array of the order(s) of the radial derivative(s) to display. Must be non-negative integers.
        If None, is it assumed that rho_derivative is 0 for all polynomials.

    theta_derivative : Optional[Union[Sequence[Integral], numpy.array]], optional
        A sequence (List, Tuple) or 1D numpy array of the order(s) of the angular derivative(s) to display. Must be non-negative integers.
        If None, is it assumed that theta_derivative is 0 for all polynomials.

    precompute : bool, optional
        If True, the useful terms for the Zernike polynomials are precomputed to optimize the computation.
        If False, the useful terms are computed on-the-fly to avoid memory overhead.
        Default is True.

    float_type : numpy.floating, optional
        The floating point type to use for the computations. Default is numpy.float64.

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
    # Get the corresponding integer types
    int_type = core_corresponding_signed_integer_type(float_type)

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
    if not isinstance(precompute, bool):
        raise TypeError("precompute must be a boolean.")
    
    # Convert n, m and rho_derivative to arrays for length checking
    n = numpy.asarray(n, dtype=int_type)
    m = numpy.asarray(m, dtype=int_type)
    if rho_derivative is not None:
        rho_derivative = numpy.asarray(rho_derivative, dtype=int_type)
    else:
        rho_derivative = numpy.zeros_like(n, dtype=int_type)
    if theta_derivative is not None:
        theta_derivative = numpy.asarray(theta_derivative, dtype=int_type)
    else:
        theta_derivative = numpy.zeros_like(n, dtype=int_type)
    
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
        raise ValueError("n, m, rho_derivative and theta_derivative (if given) must have the same length.")

    # Compute the radial polynomials using the core_polynomial function
    core_display(
        n=n,
        m=m,
        rho_derivative=rho_derivative,
        theta_derivative=theta_derivative,
        flag_radial=False,
        precompute=precompute,
        float_type=float_type,
    )
