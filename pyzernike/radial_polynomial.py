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

import numpy
from numbers import Integral, Real
from typing import Sequence, List, Optional

from .core_polynomial import core_polynomial

def radial_polynomial(
    rho: numpy.ndarray, 
    n: Sequence[Integral],
    m: Sequence[Integral],
    rho_derivative: Optional[Sequence[Integral]] = None,
    default: Real = numpy.nan,
    _skip: bool = False
) -> List[numpy.ndarray]:
    r"""
    Computes the radial Zernike polynomial :math:`R_{n}^{m}(\rho)` for :math:`\rho \leq 1`.

    The radial Zernike polynomial is defined as follows:

    .. math::

        R_{n}^{m}(\rho) = \sum_{k=0}^{(n-m)/2} \frac{(-1)^k (n-k)!}{k! ((n+m)/2 - k)! ((n-m)/2 - k)!} \rho^{n-2k}

    if :math:`n < 0`, :math:`m < 0`, :math:`n < m`, or :math:`(n - m)` is odd, the output is a zeros array with the same shape as :math:`\rho`.

    The derivative of order (derivative (a)) of the radial Zernike polynomial is defined as follows :

    .. math::

        \frac{d^{a}R_{n}^{m}(\rho)}{d\rho^{a}} = \sum_{k=0}^{(n-m)/2} \frac{(-1)^k (n-k)!}{k! ((n+m)/2 - k)! ((n-m)/2 - k)!} (n-2k) (n-2k-1) \ldots (n-2k-a+1) \rho^{n-2k-a}

    The computation of the factorial is done using the function :func:`scipy.special.gammaln` for better performance and stability.

    .. math::

        \text{log}(n!) = \text{gammaln}(n+1)

    So the coefficient of the radial polynomial is computed as follows:

    .. math::

        \frac{(n-k)!}{k! ((n+m)/2 - k)! ((n-m)/2 - k)!} = \text{exp} (\text{gammaln}(n-k+1) - \text{gammaln}(k+1) - \text{gammaln}((n+m)/2 - k + 1) - \text{gammaln}((n-m)/2 - k + 1))

    if :math:`\rho` is not in :math:`0 \leq \rho \leq 1` or :math:`\rho` is numpy.nan, the output is set to the default value (numpy.nan by default).
    
    This function allows to compute several radial Zernike polynomials at once for different orders and degrees, which can be more efficient than calling the radial polynomial function multiple times.
    The :math:`\rho` values are the same for all the polynomials, and the orders and degrees are provided as sequences. The output is a list of numpy arrays, each containing the values of the radial Zernike polynomial for the corresponding order and degree.

    .. note::

        The alias ``R`` is available for this function.

    .. note::

        For developers, the ``_skip`` parameter is used to skip the checks for the input parameters. This is useful for internal use where the checks are already done.
        In this case :

        - ``rho`` must be numpy.ndarray in the valid domain with finite values and float64 dtype.
        - ``n``, ``m`` and ``rho_derivative`` must be given as sequence of integers with the same length and valid values.

        See also :func:`pyzernike.core_polynomial` for more details on the input parameters.

    .. seealso::

        :func:`pyzernike.zernike_polynomial` for computing the Zernike polynomial :math:`Z_{n}^{m}(\rho, \theta)`.

    Parameters
    ----------
    rho : numpy.ndarray (N-D array)
        The radial coordinate values with shape (...,).

    n : Sequence[Integral]
        A list of the radial order(s) of the Zernike polynomial(s) to compute.

    m : Sequence[Integral]
        A list of the radial degree(s) of the Zernike polynomial(s) to compute.

    rho_derivative : Optional[Sequence[Integral]], optional
        A list of the order(s) of the radial derivative(s) to compute.
        If None, is it assumed that rho_derivative is 0 for all polynomials.

    default : Real, optional
        The default value for invalid rho values. The default is numpy.nan.
        If the radial coordinate values are not in the valid domain (0 <= rho <= 1) or if they are numpy.nan, the output is set to this value.

    _skip : bool, optional
        If True, the checks for the input parameters are skipped. This is useful for internal use where the checks are already done.
        The default is False.

    Returns
    -------
    List[numpy.ndarray]
        A list of numpy arrays containing the radial Zernike polynomial values for each order and degree.
        Each array has the same shape as ``rho``.

    Raises
    ------
    TypeError
        If the rho values are not a numpy array or if n and m are not integers.

    ValueError
        If the lengths of n and m are not the same.

    Examples
    --------
    Compute the radial Zernike polynomial :math:`R_{2}^{0}(\rho)` for :math:`\rho \leq 1`:

    .. code-block:: python

        import numpy
        from pyzernike import radial_polynomial
        rho = numpy.linspace(0, 1, 100)
        result = radial_polynomial(rho, n=[2], m=[0])
        polynomial = result[0]  # result is a list, we take the first element

    Compute the radial Zernike polynomial :math:`R_{2}^{0}(\rho)` and its first derivative for :math:`\rho \leq 1`:

    .. code-block:: python

        import numpy
        from pyzernike import radial_polynomial
        rho = numpy.linspace(0, 1, 100)
        result = radial_polynomial(rho, n=[2,2], m=[0,0], rho_derivative=[0, 1])
        polynomial = result[0]  # result is a list, we take the first element
        derivative = result[1]  # result is a list, we take the second element

    """
    if not _skip:
        rho = numpy.asarray(rho, dtype=numpy.float64)
        if not isinstance(n, Sequence) or not all(isinstance(i, Integral) for i in n):
            raise TypeError("n must be a sequence of integers.")
        if not isinstance(m, Sequence) or not all(isinstance(i, Integral) for i in m):
            raise TypeError("m must be a sequence of integers.")
        if rho_derivative is not None:
            if not isinstance(rho_derivative, Sequence) or not all(isinstance(i, Integral) and i >= 0 for i in rho_derivative):
                raise TypeError("rho_derivative must be a sequence of non-negative integers.")
        if not isinstance(default, Real):
            raise TypeError("Default value must be a real number.")
        
        if len(n) != len(m):
            raise ValueError("n and m must have the same length.")
        if rho_derivative is not None and len(n) != len(rho_derivative):
            raise ValueError("n and rho_derivative must have the same length.")
        if rho_derivative is None:
            rho_derivative = [0] * len(n)

        # Compute the Mask for valid rho values
        domain_mask = (rho >= 0) & (rho <= 1)
        finite_mask = numpy.isfinite(rho)
        valid_mask = domain_mask & finite_mask

        # Conserve only the valid values and save the input shape
        original_shape = rho.shape
        rho = rho[valid_mask]

    # Compute the radial polynomials using the core_polynomial function
    radial_polynomials = core_polynomial(
        rho=rho,
        theta=None,
        n=n,
        m=m,
        rho_derivative=rho_derivative,
        theta_derivative=None,
        flag_radial=True
    )

    # If rho is not in the valid domain, set the output to the default value
    if not _skip:
        for index in range(len(radial_polynomials)):
            # Reshape the radial polynomial to the original shape of rho and set the invalid values to the default value
            output_default = numpy.full(original_shape, default, dtype=numpy.float64)
            output_default[valid_mask] = radial_polynomials[index]
            radial_polynomials[index] = output_default

    # Return the radial polynomials
    return radial_polynomials
