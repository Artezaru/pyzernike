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
from typing import Sequence, List, Optional, Union
from numbers import Integral, Real

from .core.core_polynomial import core_polynomial
from .zernike_index_to_order import zernike_index_to_order
from .core.core_corresponding_signed_integer_type import core_corresponding_signed_integer_type

def zernike_polynomial_up_to_order(
        rho: numpy.ndarray,
        theta: numpy.ndarray,
        order: Integral,
        rho_derivative: Optional[Union[numpy.array, Sequence[Integral]]] = None,
        theta_derivative: Optional[Union[numpy.array, Sequence[Integral]]] = None,
        default: Real = numpy.nan,
        precompute: bool = True,
    ) -> List[List[numpy.ndarray]]:
    r"""
    Computes all the Zernike polynomials :math:`Z_n^m` for :math:`\rho \leq 1` and :math:`\theta \in [0, 2\pi]` up to a given order.

    The Zernike polynomial is defined as follows:

    .. math::

        Z_{n}^{m}(\rho, \theta) = R_{n}^{m}(\rho) \cos(m \theta) \quad \text{if} \quad m \geq 0

    .. math::

        Z_{n}^{m}(\rho, \theta) = R_{n}^{-m}(\rho) \sin(-m \theta) \quad \text{if} \quad m < 0

    If :math:`\rho` is not in :math:`0 \leq \rho \leq 1` or :math:`\rho` is numpy.nan, the output is set to the default value (numpy.nan by default).

    .. seealso::

        - :func:`pyzernike.zernike_polynomial` to compute a sets of Zernike polynomial for given order and azimuthal frequency.
        - :func:`pyzernike.zernike_index_to_order` to extract the Zernike orders (n, m) from the indices (j) in OSA/ANSI ordering.
        - The page :doc:`../../mathematical_description` in the documentation for the detailed mathematical description of the Zernike polynomials.

    This function allows to compute Zernike polynomials at once for different sets of derivative orders given as sequences,
    which can be more efficient than calling the function multiple times for each set of derivative orders.

    - The parameters ``rho`` and ``theta`` must be numpy arrays of the same shape.
    - The parameters ``rho_derivative`` and ``theta_derivative`` must be sequences of integers with the same length.

    The :math:`\rho` and :math:`\theta` values are the same for all the polynomials.
    The output ``output[k][j]`` is the Zernike polynomial of order ``n[j]`` and azimuthal frequency ``m[j]`` (OSA/ANSI ordering) with same shape as ``rho`` and for the radial derivative of order ``rho_derivative[k]`` and the angular derivative of order ``theta_derivative[k]``.

    .. note::

        If the input ``rho`` or ``theta`` are not floating point numpy arrays, it is converted to one with ``numpy.float64`` dtype.
        If the input ``rho`` or ``theta`` are floating point numpy arrays (ex: ``numpy.float32``), the computation will be done in ``numpy.float32``.
        If the input ``rho`` and ``theta`` are not of the same dtype, they are both converted to ``numpy.float64``.

    Parameters
    ----------
    rho : numpy.ndarray (N-D array)
        The radial coordinate values with shape (...,) and float64 dtype.

    theta : numpy.ndarray (N-D array)
        The angular coordinate values with shape (...,) and float64 dtype with same shape as `rho`.
    
    order : int
        The maximum order of the Zernike polynomials to compute. It must be a positive integer.

    rho_derivative : Optional[Union[Sequence[Integral], numpy.array]], optional
        A sequence (List, Tuple) or 1D numpy array of the order(s) of the radial derivative(s) to compute. Must be non-negative integers.
        If None, is it assumed that rho_derivative is 0 for all polynomials.

    theta_derivative : Optional[Union[Sequence[Integral], numpy.array]], optional
        A sequence (List, Tuple) or 1D numpy array of the order(s) of the angular derivative(s) to compute. Must be non-negative integers.
        If None, is it assumed that theta_derivative is 0 for all polynomials.

    default : Real, optional
        The default value for invalid rho values. The default is numpy.nan.
        If the radial coordinate values are not in the valid domain (0 <= rho <= 1) or if they are numpy.nan, the output is set to this value.

    precompute : bool, optional
        If True, precomputes the useful terms for better performance when computing multiple polynomials with the same rho values.
        This can significantly speed up the computation, especially for high-order polynomials.
        If False, the function will compute the terms on-the-fly, which may be slower but avoid memory overhead.
        The default is True.

    Returns
    -------
    List[List[numpy.ndarray]]
        A list of lists of numpy arrays, where each inner list corresponds to a different radial order and contains the computed Zernike polynomials for the specified orders and azimuthal frequencies.
        The shape of each array is the same as the input `rho` and `theta`, and the dtype is float64.
        ``output[k][j]`` is the Zernike polynomial of order ``n[j]`` and azimuthal frequency ``m[j]`` (OSA/ANSI ordering) with the radial derivative of order ``rho_derivative[k]`` and the angular derivative of order ``theta_derivative[k]``.

    Raises
    ------
    TypeError
        If the rho or theta values can not be converted to a numpy array of floating points values.
        If rho_derivative or theta_derivative (if not None) are not sequences of integers.

    ValueError
        If the rho and theta do not have the same shape.
        If the lengths of rho_derivative and theta_derivative (if not None) are not the same.

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
        n, m = zernike_index_to_order(indices)  # Get the orders and azimuthal frequencies from the indices

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

    The output will contain the Zernike polynomials and their derivatives for the specified orders and azimuthal frequencies.
    
    """
    # Convert rho and theta to numpy arrays of floating point values
    if not isinstance(rho, numpy.ndarray):
        rho = numpy.asarray(rho, dtype=numpy.float64)
    if not isinstance(theta, numpy.ndarray):
        theta = numpy.asarray(theta, dtype=numpy.float64)

    # Convert rho and theta in arrays of floating point values if they are not already
    if not numpy.issubdtype(rho.dtype, numpy.floating):
        rho = rho.astype(numpy.float64)
    if not numpy.issubdtype(theta.dtype, numpy.floating):
        theta = theta.astype(numpy.float64)

    # If rho and theta are not of the same dtype, convert them to float64
    if rho.dtype != theta.dtype:
        theta = theta.astype(numpy.float64)
        rho = rho.astype(numpy.float64)

    # Check that rho and theta have the same shape
    if rho.shape != theta.shape:
        raise ValueError("rho and theta must have the same shape.")

    # Determine the float type
    float_type = rho.dtype.type
    int_type = core_corresponding_signed_integer_type(float_type)

    # Check the input parameters
    if not isinstance(order, Integral) or order < 0:
        raise TypeError("order must be a non-negative integer.")
    if rho_derivative is not None:
        if not isinstance(rho_derivative, (Sequence, numpy.ndarray)) or not all(isinstance(i, Integral) and i >= 0 for i in rho_derivative):
            raise TypeError("rho_derivative must be a sequence or a 1D array of non-negative integers.")
    if theta_derivative is not None:
        if not isinstance(theta_derivative, (Sequence, numpy.ndarray)) or not all(isinstance(i, Integral) and i >= 0 for i in theta_derivative):
            raise TypeError("theta_derivative must be a sequence or a 1D array of non-negative integers.")
    if not isinstance(default, Real):
        raise TypeError("Default value must be a real number.")
    if not isinstance(precompute, bool):
        raise TypeError("precompute must be a boolean.")
    
    # Convert rho_derivative to arrays for length checking
    if rho_derivative is None and theta_derivative is None:
        rho_derivative = numpy.array([0], dtype=int_type)
        theta_derivative = numpy.array([0], dtype=int_type)
    elif rho_derivative is None and theta_derivative is not None:
        rho_derivative = numpy.zeros_like(theta_derivative, dtype=int_type)
        theta_derivative = numpy.asarray(theta_derivative, dtype=int_type)
    elif rho_derivative is not None and theta_derivative is None:
        theta_derivative = numpy.zeros_like(rho_derivative, dtype=int_type)
        rho_derivative = numpy.asarray(rho_derivative, dtype=int_type)
    else:
        rho_derivative = numpy.asarray(rho_derivative, dtype=int_type)
        theta_derivative = numpy.asarray(theta_derivative, dtype=int_type)

    # Check lengths
    if rho_derivative.ndim != 1:
        raise TypeError("rho_derivative must be a sequence or a 1D array of integers.")
    if theta_derivative.ndim != 1:
        raise TypeError("theta_derivative must be a sequence or a 1D array of integers.")

    if not (rho_derivative.size == theta_derivative.size):
        raise ValueError("rho_derivative (if given) and theta_derivative (if given) must have the same length.")
    
    # Convert the default value to the proper float type
    default = float_type(default)

    # Compute the Mask for valid rho values
    domain_mask = (rho >= 0) & (rho <= 1)
    finite_mask = numpy.isfinite(rho) & numpy.isfinite(theta)
    valid_mask = domain_mask & finite_mask

    # Conserve only the valid values and save the input shape
    original_shape = rho.shape
    rho = rho[valid_mask]
    theta = theta[valid_mask]
    
    # Convert the order into the correct int type
    order = int_type(order)

    # Create the [n,m,...] lists for all the Zernike polynomials up to the given order
    N_polynomials = (order + 1) * (order + 2) // 2
    N_derivatives = len(rho_derivative)
    n, m = zernike_index_to_order(list(range(N_polynomials)))

    # Extend n and m to match the length of rho_derivative and theta_derivative
    n = n * len(rho_derivative)
    m = m * len(rho_derivative)
    rho_derivative = [dr for dr in rho_derivative for _ in range(N_polynomials)]
    theta_derivative = [dt for dt in theta_derivative for _ in range(N_polynomials)]

    n = numpy.asarray(n, dtype=int_type)
    m = numpy.asarray(m, dtype=int_type)
    rho_derivative = numpy.asarray(rho_derivative, dtype=int_type)
    theta_derivative = numpy.asarray(theta_derivative, dtype=int_type)

    # Compute the Zernike polynomials using the core_polynomial function
    output = core_polynomial(
        rho=rho,
        theta=theta,
        n=n,
        m=m,
        rho_derivative=rho_derivative,
        theta_derivative=theta_derivative,
        flag_radial=False,
        precompute=precompute,
        float_type=float_type,
    ) # List[N_polys * len(rho_derivative) of numpy.ndarray with shape of valid rho]

    # Reshape the output to a list of lists of shape (len(rho_derivative), N_polynomials)
    output = [output[i * N_polynomials:(i + 1) * N_polynomials] for i in range(N_derivatives)]

    # =================================================================
    # Reshape the output to the original shape of rho and set the invalid values to the default value
    # =================================================================
    # If rho is not in the valid domain, set the output to the default value
    for derivative_index in range(N_derivatives):
        for index in range(N_polynomials):
            # Reshape the radial polynomial to the original shape of rho and set the invalid values to the default value
            output_default = numpy.full(original_shape, default, dtype=float_type)
            output_default[valid_mask] = output[derivative_index][index]
            output[derivative_index][index] = output_default

    return output


