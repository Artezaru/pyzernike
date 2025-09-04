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
from typing import Sequence, List, Optional
from numbers import Integral, Real
from scipy.special import gammaln

def zernike_polynomial_up_to_order(
        rho: numpy.ndarray,
        theta: numpy.ndarray,
        order: Integral,
        rho_derivative: Optional[Sequence[Integral]] = None, 
        theta_derivative: Optional[Sequence[Integral]] = None,
        default: Real = numpy.nan,
        _skip: bool = False,
    ) -> List[List[numpy.ndarray]]:
    r"""
    This method computes all the Zernike polynomials up to a given order for different orders and degrees, including their derivatives with respect to the radial and angular coordinates.

    ``output[k][j]`` is the Zernike polynomial of order ``n[j]`` and degree ``m[j]`` (OSA/ANSI ordering) with the radial derivative of order ``rho_derivative[k]`` and the angular derivative of order ``theta_derivative[k]``.

    This method is more optimized than the :func:`pyzernike.core_polynomial` method, as it assemblate the polynomials boucling on the radial parts and not the orders and degrees to avoid recomputing the same radial parts multiple times.

    .. seealso::

        - :func:`pyzernike.core_polynomial` for computing selected Zernike polynomial :math:`Z_{n}^{m}(\rho, \theta)` or the radial Zernike polynomial :math:`R_{n}^{m}(\rho)`.
        - :func:`pyzernike.zernike_index_to_order` to extract the Zernike orders (n, m) from the indices.

    Assemble the Zernike polynomial :math:`Z_{n}^{m}(\rho, \theta)`.

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

    The computation of the factorial is done using the function :func:`scipy.special.gammaln` for better performance and stability.

    .. math::

        \text{log}(n!) = \text{gammaln}(n+1)

    So the coefficient of the radial polynomial is computed as follows:

    .. math::

        \frac{(n-k)!}{k! ((n+m)/2 - k)! ((n-m)/2 - k)!} = \text{exp} (\text{gammaln}(n-k+1) - \text{gammaln}(k+1) - \text{gammaln}((n+m)/2 - k + 1) - \text{gammaln}((n-m)/2 - k + 1))

    For points that not respect the condition :math:`0 <= \rho <= 1`, the output is set to the `default` value (default is `numpy.nan`).

    .. note::

        The fonction is designed to precompute the useful terms for the Zernike polynomials, such as the powers of rho, the cosine and sine terms, and the logarithm of the factorials.

    .. note::

        An alias for this function is ``Zup``.
    
    .. note::

        For developers, the ``_skip`` parameter is used to skip the checks for the input parameters. This is useful for internal use where the checks are already done.
        In this case :

        - ``rho`` and ``theta`` must be numpy.ndarray in the valid domain with finite values and float64 dtype and the same shape.
        - ``rho_derivative`` and ``theta_derivative`` must be given as sequence of integers with the same length and valid values.

    Parameters
    ----------
    rho : numpy.ndarray (N-D array)
        The radial coordinate values with shape (...,) and float64 dtype.

    theta : numpy.ndarray (N-D array)
        The angular coordinate values with shape (...,) and float64 dtype with same shape as `rho`.
    
    order : int
        The maximum order of the Zernike polynomials to compute. It must be a positive integer.

    rho_derivative : Sequence[int]
        A list of integers containing the order of the radial derivative to compute for each radial Zernike polynomial.
        If `rho_derivative` is None, no radial derivative is computed. Assuming that the radial derivative is 0 for all polynomials.

    theta_derivative : Sequence[int]
        A list of integers containing the order of the angular derivative to compute for each Zernike polynomial. Same length as ``rho_derivative``.
        If `theta_derivative` is None, no angular derivative is computed. Assuming that the angular derivative is 0 for all polynomials.

    default : Real, optional
        The default value for invalid rho values. The default is numpy.nan.
        If the radial coordinate values are not in the valid domain (0 <= rho <= 1) or if they are numpy.nan, the output is set to this value.

    _skip : bool, optional
        If True, skips input validation checks. Default is False. This is useful for internal use where the checks are already done.

    Returns
    -------
    List[List[numpy.ndarray]]
        A list of lists of numpy arrays, where each inner list corresponds to a different radial order and contains the computed Zernike polynomials for the specified orders and degrees.
        The shape of each array is the same as the input `rho` and `theta`, and the dtype is float64.
        ``output[k][j]`` is the Zernike polynomial of order ``n[j]`` and degree ``m[j]`` (OSA/ANSI ordering) with the radial derivative of order ``rho_derivative[k]`` and the angular derivative of order ``theta_derivative[k]``.

    Raises
    ------
    TypeError
        If `rho` or `theta` are not numpy arrays, or if `order` is not an integer, or if `rho_derivative` or `theta_derivative` are not sequences of integers.

    ValueError
        If `rho` and `theta` do not have the same shape, or if `order` is negative, or if `rho_derivative` or `theta_derivative` are not of the same length, or if `rho_derivative` or `theta_derivative` contain negative integers.

    Examples
    --------
    
    Compute all the Zernike polynomials up to order 3 for a grid of points:

    .. code-block:: python

        import numpy
        from pyzernike import zernike_polynomial_up_to_order, zernike_index_to_order

        # Create a grid of points
        rho = numpy.linspace(0, 1, 100)
        theta = numpy.linspace(0, 2 * numpy.pi, 100)

        # Compute the Zernike polynomials up to order 3
        result = zernike_polynomial_up_to_order(rho, theta, order=3)
        polynomials = result[0]  # Get the first set of polynomials (for rho_derivative=0, theta_derivative=0)

        # Extract the values: 
        indices = list(range(len(polynomials)))
        n, m = zernike_index_to_order(indices)  # Get the orders and degrees from the indices

        for i, (n_i, m_i) in enumerate(zip(n, m)):
            print(f"Zernike polynomial Z_{n_i}^{m_i} for the given rho and theta values is: {polynomials[i]}")

    To compute the polynomials and their first derivatives with respect to rho:

    .. code-block:: python

        import numpy
        from pyzernike import zernike_polynomial_up_to_order, zernike_index_to_order

        # Create a grid of points
        rho = numpy.linspace(0, 1, 100)
        theta = numpy.linspace(0, 2 * numpy.pi, 100)

        # Compute the Zernike polynomials up to order 3 with radial derivatives
        result = zernike_polynomial_up_to_order(rho, theta, order=3, rho_derivative=[0, 1], theta_derivative=[0, 0])
        polynomials = result[0]  # Get the first set of polynomials (for rho_derivative=0, theta_derivative=0)
        derivatives = result[1]  # Get the second set of polynomials (for rho_derivative=1, theta_derivative=0)

    The output will contain the Zernike polynomials and their derivatives for the specified orders and degrees.
    
    """
    if not _skip:
        rho = numpy.asarray(rho, dtype=numpy.float64)
        theta = numpy.asarray(theta, dtype=numpy.float64)
        if not isinstance(order, Integral) or order < 0:
            raise TypeError("Order must be a non-negative integer.")
        if rho_derivative is not None:
            if not isinstance(rho_derivative, Sequence) or not all(isinstance(i, Integral) and i >= 0 for i in rho_derivative):
                raise TypeError("rho_derivative must be a sequence of non-negative integers.")
        if theta_derivative is not None:
            if not isinstance(theta_derivative, Sequence) or not all(isinstance(i, Integral) and i >= 0 for i in theta_derivative):
                raise TypeError("theta_derivative must be a sequence of non-negative integers.")
        if not isinstance(default, Real):
            raise TypeError("Default value must be a real number.")
        
        if not rho.shape == theta.shape:
            raise ValueError("Rho and theta must have the same shape.")
        if rho_derivative is not None and theta_derivative is not None and len(rho_derivative) != len(theta_derivative):
            raise ValueError("rho_derivative and theta_derivative must have the same length.")
        if theta_derivative is not None and rho_derivative is None:
            rho_derivative = [0] * len(theta_derivative)
        if rho_derivative is not None and theta_derivative is None:
            theta_derivative = [0] * len(rho_derivative)
        if rho_derivative is None and theta_derivative is None:
            rho_derivative = [0]
            theta_derivative = [0]

        # Compute the Mask for valid rho values
        domain_mask = (rho >= 0) & (rho <= 1)
        finite_mask = numpy.isfinite(rho) & numpy.isfinite(theta)
        valid_mask = domain_mask & finite_mask

        # Conserve only the valid values and save the input shape
        original_shape = rho.shape
        rho = rho[valid_mask]
        theta = theta[valid_mask]

    # Create the output list
    max_index = (order * (order + 3)) // 2
    output = [[None for _ in range(max_index + 1)] for _ in range(len(rho_derivative))]

    # =================================================================
    # Precomputation of the useful terms
    # =================================================================
    # rho_powers : numpy.ndarray (2-D array)
    #     An array of shape=(..., order + 1) containing the precomputed powers of rho for the useful exponents from 0 to order + 1.
    # 
    # cosine_terms : numpy.ndarray (2-D array)
    #     An array of shape=(..., order + 1) containing the cosine terms for the useful angular polynomials.
    # 
    # sine_terms : numpy.ndarray (2-D array)
    #     An array of shape=(..., order + 1) containing the sine terms for the useful angular polynomials.
    # 
    # log_factorials : numpy.ndarray (1-D array)
    #     An array of shape=(order + 1,) containing the logarithm of the factorials for the useful integers from 0 to order + 1.
    # =================================================================

    # Construct the precomputed terms
    rho_powers = numpy.power(rho[..., numpy.newaxis], numpy.arange(0, order + 1, dtype=int))
    log_factorials = gammaln(numpy.arange(0, order + 1, dtype=int) + 1)
    cosine_terms = numpy.cos(numpy.arange(0, order + 1, dtype=int) * theta[..., numpy.newaxis])
    sine_terms = numpy.sin(numpy.arange(0, order + 1, dtype=int) * theta[..., numpy.newaxis])

    # =================================================================
    # Uniquing the indices of the rho_derivative and theta_derivative
    # =================================================================
    # For a same rho_derivative, the values of the radial polynomials are the same.
    # For a same theta_derivative, the values of the angular polynomials are the same.
    #
    # So we compute the radial and the angular parts only onces for each unique rho_derivative and theta_derivative.
    # Then a simple multiplication is done to compute the Zernike polynomial for each selected order and degree.
    # =================================================================

    unique_rho_derivative = numpy.unique(rho_derivative)
    unique_theta_derivative = numpy.unique(theta_derivative)

    # ==============================================================
    # Precompute the angular polynomial for the current m for each unique theta_derivative
    # ==============================================================
    # As the angular polynomial is independent of n, we can precompute it only once for each unique theta_derivative before the loop over n and m.
    #
    # We compute the cosine and sine terms for each unique theta_derivative.
    #
    # cosine_precomputed[(m, theta_derivative_idx)] = access to the cosine term for the given m and theta_derivative_idx
    # sine_precomputed[(m, theta_derivative_idx)] = access to the sine term for the given m and theta_derivative_idx
    # =================================================================
    cosine_precomputed = {}
    sine_precomputed = {}
    for m in range(order + 1):
        for theta_derivative_idx in unique_theta_derivative:
            # According to the angular derivative, we compute the cosine and sine factors
            if m == 0:
                if theta_derivative_idx == 0:
                    # For m = 0, only the cosine term is non-zero
                    cosine_precomputed[(0, theta_derivative_idx)] = 1.0
                else:
                    cosine_precomputed[(0, theta_derivative_idx)] = 0.0
            elif m > 0:
                if theta_derivative_idx == 0:
                    cosine_precomputed[(m, theta_derivative_idx)] = cosine_terms[..., m]
                    sine_precomputed[(m, theta_derivative_idx)] = sine_terms[..., m]
                elif theta_derivative_idx % 4 == 0:
                    cosine_precomputed[(m, theta_derivative_idx)] = (m ** theta_derivative_idx) * cosine_terms[..., m]
                    sine_precomputed[(m, theta_derivative_idx)] = (m ** theta_derivative_idx) * sine_terms[..., m]
                elif theta_derivative_idx % 4 == 1:
                    cosine_precomputed[(m, theta_derivative_idx)] = - (m ** theta_derivative_idx) * sine_terms[..., m]
                    sine_precomputed[(m, theta_derivative_idx)] = (m ** theta_derivative_idx) * cosine_terms[..., m]
                elif theta_derivative_idx % 4 == 2:
                    cosine_precomputed[(m, theta_derivative_idx)] = - (m ** theta_derivative_idx) * cosine_terms[..., m]
                    sine_precomputed[(m, theta_derivative_idx)] = - (m ** theta_derivative_idx) * sine_terms[..., m]
                else:
                    cosine_precomputed[(m, theta_derivative_idx)] = (m ** theta_derivative_idx) * sine_terms[..., m]
                    sine_precomputed[(m, theta_derivative_idx)] = - (m ** theta_derivative_idx) * cosine_terms[..., m]
    
    # =================================================================
    # Boucle over the polynomials to compute
    # =================================================================
    for n in range(order + 1):
        # We boucle only over the positive m values, as the negative m values are computed by symmetry.
        start_m = 0 if n % 2 == 0 else 1
        for m in range(start_m, n + 1, 2):

            # ==============================================================
            # Precompute the radial polynomial for the current n and m for each unique rho_derivative
            # ==============================================================
            # For a same rho_derivative, the values of the radial polynomials are the same.
            # So we compute the radial polynomial only once for each unique rho_derivative before the loop
            #
            # radials_precomputed[rho_derivative_idx] = access to the radial polynomial for the given rho_derivative_idx (and the current n and m)
            # =================================================================
            radials_precomputed = {}
            for rho_derivative_idx in unique_rho_derivative:
                # Compute the number of terms
                s = min((n - m) // 2, (n - rho_derivative_idx) // 2) # No computation for terms derivated more than the index of the polynomial
                k = numpy.arange(0, s + 1)

                # Compute the coefficients
                log_k_coef = log_factorials[n - k] - \
                            log_factorials[k] - \
                            log_factorials[(n + m) // 2 - k] - \
                            log_factorials[(n - m) // 2 - k]

                sign = 1 - 2 * (k % 2)

                if rho_derivative_idx != 0:
                    log_k_coef += log_factorials[n - 2 * k] - log_factorials[n - 2 * k - rho_derivative_idx]

                coef = sign * numpy.exp(log_k_coef)

                # Compute the rho power
                exponent = n - 2 * k - rho_derivative_idx
                rho_orders = rho_powers[..., exponent]

                # Assemble the radial polynomial if flag_radial is True
                radials_precomputed[rho_derivative_idx] = numpy.tensordot(rho_orders, coef, axes=[[-1], [0]])
                
            # ==============================================================
            # Boucle over the rho_derivative and theta_derivative to assemblate the Zernike polynomial
            # ==============================================================
            for derivative_index in range(len(rho_derivative)):
                rho_derivative_idx = rho_derivative[derivative_index]
                theta_derivative_idx = theta_derivative[derivative_index]

                # Get the radial polynomial for the current rho_derivative
                result = radials_precomputed[rho_derivative_idx]

                # Check if the m is positive or negative
                if m == 0:
                    cosine_result = result * cosine_precomputed[(0, theta_derivative_idx)]
                    output[derivative_index][n*(n+2)//2] = cosine_result
                elif m > 0:
                    cosine_result = result * cosine_precomputed[(m, theta_derivative_idx)]
                    output[derivative_index][(n*(n+2) + m) // 2] = cosine_result
                    sine_result = result * sine_precomputed[(m, theta_derivative_idx)]
                    output[derivative_index][(n*(n+2) - m) // 2] = sine_result

    # =================================================================
    # Reshape the output to the original shape of rho and set the invalid values to the default value
    # =================================================================
    # If rho is not in the valid domain, set the output to the default value
    if not _skip:
        for derivative_index in range(len(output)):
            for index in range(len(output[derivative_index])):
                # Reshape the radial polynomial to the original shape of rho and set the invalid values to the default value
                output_default = numpy.full(original_shape, default, dtype=numpy.float64)
                output_default[valid_mask] = output[derivative_index][index]
                output[derivative_index][index] = output_default

    return output


