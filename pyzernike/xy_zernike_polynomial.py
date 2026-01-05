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
from typing import Sequence, List, Optional, Union

from .core.core_polynomial import core_polynomial
from .core.core_cartesian_to_elliptic_annulus import core_cartesian_to_elliptic_annulus
from .core.core_corresponding_signed_integer_type import core_corresponding_signed_integer_type

def xy_zernike_polynomial(
    x: numpy.ndarray,
    y: numpy.ndarray,
    n: Union[numpy.array, Sequence[Integral]],
    m: Union[numpy.array, Sequence[Integral]],
    Rx: Real = 1.0, 
    Ry: Real = 1.0, 
    x0: Real = 0.0, 
    y0: Real = 0.0, 
    alpha: Real = 0.0, 
    h: Real = 0.0, 
    x_derivative:  Optional[Union[numpy.array, Sequence[Integral]]] = None,
    y_derivative:  Optional[Union[numpy.array, Sequence[Integral]]] = None,
    default: Real = numpy.nan,
    precompute: bool = True,
) -> List[numpy.ndarray]:
    r"""
    Computes the Zernike polynomial :math:`Z_{n}^{m}(\rho_{eq}, \theta_{eq})` for given cartesian coordinates :math:`(x, y)` on an elliptic annulus domain.

    .. seealso::

        - :func:`pyzernike.zernike_polynomial` for computing the Zernike polynomial :math:`Z_{n}^{m}` for polar coordinates :math:`(\rho, \theta)`. 
        - :func:`pyzernike.cartesian_to_elliptic_annulus` to convert Cartesian coordinates to elliptic annulus domain polar coordinates.
        - The page :doc:`../../mathematical_description` in the documentation for the mathematical extension of the Zernike polynomials on the elliptic domain.

    .. seealso::

        For the mathematical development of the method, see the paper `Generalization of Zernike polynomials for regular portions of circles and ellipses` by Rafael Navarro, José L. López, José Rx. Díaz, and Ester Pérez Sinusía.
        The associated paper is available in the resources folder of the package.

        Download the PDF : :download:`PDF <../../../pyzernike/resources/Navarro and al. Generalization of Zernike polynomials for regular portions of circles and ellipses.pdf>`

    Lets consider the extended elliptic annulus domain defined by the following parameters:

    .. figure:: ../../../pyzernike/resources/elliptic_annulus_domain.png
        :width: 400px
        :align: center

        The parameters to define the extended domain of the Zernike polynomial.

    The parameters are:

    - :math:`R_x` and :math:`R_y` are the lengths of the semi-axis of the ellipse.
    - :math:`x_0` and :math:`y_0` are the coordinates of the center of the ellipse.
    - :math:`\alpha` is the rotation angle of the ellipse in radians.
    - :math:`h=\frac{a}{R_x}=\frac{b}{R_y}` defining the inner boundary of the ellipse.

    The Zernike polynomial :math:`Z_{n}^{m}(\rho_{eq}, \theta_{eq})` is computed for the equivalent polar coordinates.
    
    This function allows to compute several Zernike polynomials at once for different sets of (order, azimuthal frequency, derivative orders) given as sequences,
    which can be more efficient than calling the polynomial function multiple times.

    - The parameters ``x`` and ``y`` must be numpy arrays of the same shape.
    - The parameters ``n``, ``m``, ``x_derivative`` and ``y_derivative`` must be sequences of integers with the same length.

    The output is a list of numpy arrays, each containing the values of the Zernike polynomial for the corresponding order and azimuthal frequency.
    The list has the same length as the input sequences and the arrays have the same shape as ``x``.

    .. note::

        If the input ``x`` or ``y`` are not floating point numpy arrays, it is converted to one with ``numpy.float64`` dtype.
        If the input ``x`` or ``y`` are floating point numpy arrays (ex: ``numpy.float32``), the computation will be done in ``numpy.float32``.
        If the input ``x`` and ``y`` are not of the same dtype, they are both converted to ``numpy.float64``.

    .. warning::

        The only available derivatives are the first and second order derivatives with respect to x or y independently (jacobian and hessian matrix).
        For more complex derivatives, please implement the Faa di Bruno's formula using the standard Zernike polynomial function with polar coordinates.

        - Available derivatives (:math:(dx, dy)): (0, 0), (1, 0), (0, 1), (2, 0), (0, 2), (1, 1).

    Parameters
    ----------
    x : numpy.ndarray
        The x coordinates in Cartesian system with shape (...,).

    y : numpy.ndarray
        The y coordinates in Cartesian system with shape (...,).

    n : Sequence[Integral] or numpy.array
        A sequence (List, Tuple) or 1D numpy array of the radial order(s) of the Zernike polynomial(s) to compute. Must be non-negative integers.

    m : Sequence[Integral] or numpy.array
        A sequence (List, Tuple) or 1D numpy array of the azimuthal frequency(ies) of the Zernike polynomial(s) to compute. Must be non-negative integers.

    Rx : Real, optional
        The length of the semi-axis of the ellipse along x axis. Must be strictly positive.
        The default is 1.0, which corresponds to the unit circle.

    Ry : Real, optional
        The length of the semi-axis of the ellipse along y axis. Must be strictly positive.
        The default is 1.0, which corresponds to the unit circle.

    x0 : Real, optional
        The x coordinate of the center of the ellipse. Can be any real number.
        The default is 0.0, which corresponds to an ellipse centered at the origin.

    y0 : Real, optional
        The y coordinate of the center of the ellipse. Can be any real number.
        The default is 0.0, which corresponds to an ellipse centered at the origin.

    alpha : Real, optional
        The rotation angle of the ellipse in radians. Can be any real number.
        The default is 0.0, such as :math:`x` and :math:`y` axis are aligned with the ellipse axes.

    h : Real, optional
        The ratio of the inner semi-axis to the outer semi-axis. Must be in the range [0, 1).
        The default is 0.0, which corresponds to a filled ellipse.

    x_derivative : Optional[Union[Sequence[Integral], numpy.array]], optional
        A sequence (List, Tuple) or 1D numpy array of the order(s) of the x derivative(s) to compute. Must be non-negative integers.
        If None, is it assumed that x_derivative is 0 for all polynomials.

    y_derivative : Optional[Union[Sequence[Integral], numpy.array]], optional
        A sequence (List, Tuple) or 1D numpy array of the order(s) of the y derivative(s) to compute. Must be non-negative integers.
        If None, is it assumed that y_derivative is 0 for all polynomials.

    default : Real, optional
        The default value to use for points outside the elliptic annulus domain. Must be a real number.
        The default is numpy.nan.

    precompute : bool, optional
        If True, precomputes the useful terms for better performance when computing multiple polynomials with the same rho values.
        If False, computes the useful terms on the fly for each polynomial to avoid memory overhead.
        The default is True.


    Returns
    -------
    List[numpy.ndarray]
        A list of numpy arrays containing the Zernike polynomial values for each order and azimuthal frequency.
        Each array has the same shape as ``x``.

    Raises
    ------
    TypeError
        If the x or y values can not be converted to a numpy array of floating points values.
        If n, m, x_derivative or y_derivative (if not None) are not sequences of integers.

    ValueError
        If the x and y do not have the same shape.
        If the lengths of n, m, x_derivative and y_derivative (if not None) are not the same.
        If Rx or Ry are not strictly positive.
        If h is not in the range [0, 1[.
        If the derivative orders are higher than 2 or mixed (ex: (1, 1)).

    Examples
    --------
    Let's consider a full circle with a radius of 10 centered at the origin (1, 1).
    The value of the zernike polynomial :math:`Z_{2}^{0}` at the point (x, y) is given by:

    .. code-block:: python

        import numpy
        from pyzernike import xy_zernike_polynomial # or Zxy
        x = numpy.linspace(-10, 10, 100)
        y = numpy.linspace(-10, 10, 100)
        X, Y = numpy.meshgrid(x, y)

        output = xy_zernike_polynomial(X, Y, n=[2], m=[0], Rx=10, Ry=10, x0=1.0, y0=1.0) # List with a single numpy array
        zernike_n2_m0 = output[0] # Shape similar to X and Y

    returns a list with a single numpy array containing the values of the Zernike polynomial :math:`Z_{2}^{0}` at the points (X, Y) within the circle of radius 10.

    To compute the first derivative with respect to x and y, we can use:

    .. code-block:: python

        output = xy_zernike_polynomial(X, Y, n=[2, 2], m=[0, 0], Rx=10, Ry=10, x0=1.0, y0=1.0, x_derivative=[1, 0], y_derivative=[0, 1]) # List with two numpy arrays
        zernike_n2_m0_dx = output[0] # First derivative with respect to x
        zernike_n2_m0_dy = output[1] # First derivative with respect to y
    
    """
    # Convert x and y to numpy arrays of floating point values
    if not isinstance(x, numpy.ndarray):
        x = numpy.asarray(x, dtype=numpy.float64)
    if not isinstance(y, numpy.ndarray):
        y = numpy.asarray(y, dtype=numpy.float64)

    # Convert x and y in arrays of floating point values if they are not already
    if not numpy.issubdtype(x.dtype, numpy.floating):
        x = x.astype(numpy.float64)
    if not numpy.issubdtype(y.dtype, numpy.floating):
        y = y.astype(numpy.float64)

    # If x and y are not of the same dtype, convert them to float64
    if x.dtype != y.dtype:
        x = x.astype(numpy.float64)
        y = y.astype(numpy.float64)

    # Check that x and y have the same shape
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")

    # Determine the float type
    float_type = y.dtype.type
    int_type = core_corresponding_signed_integer_type(float_type)

    # Check the input parameters
    if not isinstance(n, (Sequence, numpy.ndarray)) or not all(isinstance(i, Integral) for i in n):
        raise TypeError("n must be a sequence or a 1D array of integers.")
    if not isinstance(m, (Sequence, numpy.ndarray)) or not all(isinstance(i, Integral) for i in m):
        raise TypeError("m must be a sequence or a 1D array of integers.")
    if x_derivative is not None:
        if not isinstance(x_derivative, (Sequence, numpy.ndarray)) or not all(isinstance(i, Integral) and i >= 0 for i in x_derivative):
            raise TypeError("x_derivative must be a sequence or a 1D array of non-negative integers.")
    if y_derivative is not None:
        if not isinstance(y_derivative, (Sequence, numpy.ndarray)) or not all(isinstance(i, Integral) and i >= 0 for i in y_derivative):
            raise TypeError("y_derivative must be a sequence or a 1D array of non-negative integers.")
    if Rx is not None:
        if not isinstance(Rx, Real) or Rx <= 0:
            raise TypeError("Rx must be a positive real number.")
    if Ry is not None:
        if not isinstance(Ry, Real) or Ry <= 0:
            raise TypeError("Ry must be a positive real number.")
    if x0 is not None:
        if not isinstance(x0, Real):
            raise TypeError("x0 must be a real number.")
    if y0 is not None:
        if not isinstance(y0, Real):
            raise TypeError("y0 must be a real number.")
    if alpha is not None:
        if not isinstance(alpha, Real):
            raise TypeError("Alpha must be a real number.")
    if h is not None:
        if not isinstance(h, Real) or not (0 <= h < 1):
            raise TypeError("h must be a real number in the range [0, 1[.")
    if not isinstance(default, Real):
        raise TypeError("Default value must be a real number.")
    if not isinstance(precompute, bool):
        raise TypeError("precompute must be a boolean.")
    
    
    # Convert rho_derivative to arrays for length checking
    n = numpy.asarray(n, dtype=int_type)
    m = numpy.asarray(m, dtype=int_type)
    if x_derivative is not None:
        x_derivative = numpy.asarray(x_derivative, dtype=int_type)
    else:
        x_derivative = numpy.zeros_like(n, dtype=int_type)
    if y_derivative is not None:
        y_derivative = numpy.asarray(y_derivative, dtype=int_type)
    else:
        y_derivative = numpy.zeros_like(n, dtype=int_type)
    
    # Check lengths
    if not n.ndim == 1:
        raise TypeError("n must be a sequence or a 1D array of integers.")
    if not m.ndim == 1:
        raise TypeError("m must be a sequence or a 1D array of integers.")
    if not x_derivative.ndim == 1:
        raise TypeError("x_derivative must be a sequence or a 1D array of integers.")
    if not y_derivative.ndim == 1:
        raise TypeError("y_derivative must be a sequence or a 1D array of integers.")
    if not (n.size == m.size ==x_derivative.size == y_derivative.size):
        raise ValueError("n, m, x_derivative (if given) and y_derivative (if given) must have the same length.")

    # Check that the derivative orders are supported
    for index in range(len(x_derivative)):
        if (x_derivative[index], y_derivative[index]) not in [(0, 0), (1, 0), (0, 1), (2, 0), (0, 2), (1, 1)]:
            raise ValueError("The function supports only the derivatives (0, 0), (1, 0), (0, 1), (2, 0), (0, 2), and (1, 1). For more complex derivatives, use the standard Zernike polynomial function with polar coordinates.")
        
    # Convert the float parameters to the correct type
    if Rx is None:
        Rx = float_type(1.0)
    else:
        Rx = float_type(Rx)

    if Ry is None:
        Ry = float_type(1.0)
    else:
        Ry = float_type(Ry)

    if x0 is None:
        x0 = float_type(0.0)
    else:
        x0 = float_type(x0)

    if y0 is None:
        y0 = float_type(0.0)
    else:
        y0 = float_type(y0)

    if alpha is None:
        alpha = float_type(0.0)
    else:
        alpha = float_type(alpha)

    if h is None:
        h = float_type(0.0)
    else:
        h = float_type(h)

    # Convert default to the correct type
    default = float_type(default)

    # Searching what is the derivatives to compute
    total_derivative_orders = numpy.array(x_derivative) + numpy.array(y_derivative)
    max_derivative_order = numpy.max(total_derivative_orders)
    min_derivative_order = numpy.min(total_derivative_orders)
    skip_0_derivative = min_derivative_order >= 1
    shift_0_derivative = 1 if skip_0_derivative else 0

    # Compute rho_eq, theta_eq on the elliptic annulus domain and their derivatives if needed
    if max_derivative_order == 0:
        # Only extract rho and theta
        list_rho_eq, list_theta_eq = core_cartesian_to_elliptic_annulus(x, y, Rx, Ry, x0, y0, alpha, h, x_derivative=numpy.array([0], dtype=int_type), y_derivative=numpy.array([0], dtype=int_type), float_type=float_type)
    elif max_derivative_order == 1:
        # Extract rho, theta and their first derivatives
        list_rho_eq, list_theta_eq = core_cartesian_to_elliptic_annulus(x, y, Rx, Ry, x0, y0, alpha, h, x_derivative=numpy.array([0, 1, 0], dtype=int_type), y_derivative=numpy.array([0, 0, 1], dtype=int_type), float_type=float_type)
    elif max_derivative_order == 2:
        # Extract rho, theta and their first and second derivatives
        list_rho_eq, list_theta_eq = core_cartesian_to_elliptic_annulus(x, y, Rx, Ry, x0, y0, alpha, h, x_derivative=numpy.array([0, 1, 0, 2, 0, 1], dtype=int_type), y_derivative=numpy.array([0, 0, 1, 0, 2, 1], dtype=int_type), float_type=float_type)
    else:
        raise ValueError("The function supports only the derivatives (0, 0), (1, 0), (0, 1), (2, 0), (0, 2), and (1, 1). For more complex derivatives, use the standard Zernike polynomial function with polar coordinates.")

    rho = list_rho_eq[0]
    theta = list_theta_eq[0]

    # Compute the Mask for valid rho values
    domain_mask = (rho >= 0) & (rho <= 1)
    finite_mask = numpy.isfinite(rho) & numpy.isfinite(theta)
    valid_mask = domain_mask & finite_mask

    # Conserve only the valid values and save the input shape
    original_shape = rho.shape
    rho = rho[valid_mask]
    theta = theta[valid_mask]

    # Compute the Zernike polynomial and its derivatives if needed
    if max_derivative_order == 0:
        # Only the polynomial is computed
        rho_derivative = theta_derivative = [0] * len(n)
        rho_derivative = numpy.array(rho_derivative, dtype=int_type)
        theta_derivative = numpy.array(theta_derivative, dtype=int_type)
        output_polynomials = core_polynomial(rho, theta, n, m, rho_derivative=rho_derivative, theta_derivative=theta_derivative, flag_radial=False, precompute=precompute, float_type=float_type)
    elif max_derivative_order == 1 and not skip_0_derivative:
        # The first derivatives of the Zernike polynomial are also computed
        rho_derivative = [0] * len(n) + [1] * len(n) + [0] * len(n)
        rho_derivative = numpy.array(rho_derivative, dtype=int_type)
        theta_derivative = [0] * len(n) + [0] * len(n) + [1] * len(n)   
        theta_derivative = numpy.array(theta_derivative, dtype=int_type)
        compute_n = numpy.concatenate([n, n, n], axis=0, dtype=int_type)
        compute_m = numpy.concatenate([m, m, m], axis=0, dtype=int_type)
        output_polynomials = core_polynomial(rho, theta, compute_n, compute_m, rho_derivative=rho_derivative, theta_derivative=theta_derivative, flag_radial=False, precompute=precompute, float_type=float_type)
    elif max_derivative_order == 1 and skip_0_derivative:
        # The first derivatives of the Zernike polynomial are also computed
        rho_derivative = [1] * len(n) + [0] * len(n)
        rho_derivative = numpy.array(rho_derivative, dtype=int_type)
        theta_derivative = [0] * len(n) + [1] * len(n)
        theta_derivative = numpy.array(theta_derivative, dtype=int_type)
        compute_n = numpy.concatenate([n, n], axis=0, dtype=int_type)
        compute_m = numpy.concatenate([m, m], axis=0, dtype=int_type)
        output_polynomials = core_polynomial(rho, theta, compute_n, compute_m, rho_derivative=rho_derivative, theta_derivative=theta_derivative, flag_radial=False, precompute=precompute, float_type=float_type)
    elif max_derivative_order == 2 and not skip_0_derivative:
        # The second derivatives of the Zernike polynomial are also computed
        rho_derivative = [0] * len(n) + [1] * len(n) + [0] * len(n) + [2] * len(n) + [0] * len(n) + [1] * len(n)
        rho_derivative = numpy.array(rho_derivative, dtype=int_type)
        theta_derivative = [0] * len(n) + [0] * len(n) + [1] * len(n) + [0] * len(n) + [2] * len(n) + [1] * len(n)
        theta_derivative = numpy.array(theta_derivative, dtype=int_type)
        compute_n = numpy.concatenate([n, n, n], axis=0, dtype=int_type)
        compute_m = numpy.concatenate([m, m, m], axis=0, dtype=int_type)
        output_polynomials = core_polynomial(rho, theta, compute_n, compute_m, rho_derivative=rho_derivative, theta_derivative=theta_derivative, flag_radial=False, precompute=precompute, float_type=float_type)
    elif max_derivative_order == 2 and skip_0_derivative:
        # The second derivatives of the Zernike polynomial are also computed
        rho_derivative = [1] * len(n) + [0] * len(n) + [2] * len(n) + [0] * len(n) + [1] * len(n)
        rho_derivative = numpy.array(rho_derivative, dtype=int_type)
        theta_derivative = [0] * len(n) + [1] * len(n) + [0] * len(n) + [2] * len(n) + [1] * len(n)
        theta_derivative = numpy.array(theta_derivative, dtype=int_type)
        compute_n = numpy.concatenate([n, n, n, n, n], axis=0, dtype=int_type)
        compute_m = numpy.concatenate([m, m, m, m, m], axis=0, dtype=int_type)
        output_polynomials = core_polynomial(rho, theta, compute_n, compute_m, rho_derivative=rho_derivative, theta_derivative=theta_derivative, flag_radial=False, precompute=precompute, float_type=float_type)
    else:
        raise ValueError("The function supports only the derivatives (0, 0), (1, 0), (0, 1), (2, 0), (0, 2), and (1, 1). For more complex derivatives, use the standard Zernike polynomial function with polar coordinates.")

    # Compose the final output depending on the derivative orders
    output = []
    for index in range(len(n)):
        # Case (0, 0) : Only extract the polynomial
        if (x_derivative[index], y_derivative[index]) == (0, 0):
            output.append(output_polynomials[index])

        # Case (1, 0) : Extract the first derivative with respect to x
        elif (x_derivative[index], y_derivative[index]) == (1, 0):
            drho_dx = list_rho_eq[1]
            dtheta_dx = list_theta_eq[1]
            dZ_drho = output_polynomials[(1 - shift_0_derivative) * len(n) + index]
            dZ_dtheta = output_polynomials[(2 - shift_0_derivative) * len(n) + index]
            dZ_dx = dZ_drho * drho_dx[valid_mask] + dZ_dtheta * dtheta_dx[valid_mask]
            output.append(dZ_dx)

        # Case (0, 1) : Extract the first derivative with respect to y
        elif (x_derivative[index], y_derivative[index]) == (0, 1):
            drho_dy = list_rho_eq[2]
            dtheta_dy = list_theta_eq[2]
            dZ_drho = output_polynomials[(1 - shift_0_derivative) * len(n) + index]
            dZ_dtheta = output_polynomials[(2 - shift_0_derivative) * len(n) + index]
            dZ_dy = dZ_drho * drho_dy[valid_mask] + dZ_dtheta * dtheta_dy[valid_mask]
            output.append(dZ_dy)

        # Case (2, 0) : Extract the second derivative with respect to x
        elif (x_derivative[index], y_derivative[index]) == (2, 0):
            drho_dx = list_rho_eq[1]
            dtheta_dx = list_theta_eq[1]
            d2rho_dx2 = list_rho_eq[3]
            d2theta_dx2 = list_theta_eq[3]
            dZ_drho = output_polynomials[(1 - shift_0_derivative) * len(n) + index]
            dZ_dtheta = output_polynomials[(2 - shift_0_derivative) * len(n) + index]
            d2Z_drho2 = output_polynomials[(3 - shift_0_derivative) * len(n) + index]
            d2Z_dtheta2 = output_polynomials[(4 - shift_0_derivative) * len(n) + index]
            d2Z_drhodtheta = output_polynomials[(5 - shift_0_derivative) * len(n) + index]
            d2Z_dx2 = (d2Z_drho2 * drho_dx[valid_mask] + d2Z_drhodtheta * dtheta_dx[valid_mask]) * drho_dx[valid_mask] + \
                      dZ_drho * d2rho_dx2[valid_mask] + \
                      (d2Z_dtheta2 * dtheta_dx[valid_mask] + d2Z_drhodtheta * drho_dx[valid_mask]) * dtheta_dx[valid_mask] + \
                      dZ_dtheta * d2theta_dx2[valid_mask]
            output.append(d2Z_dx2)

        # Case (0, 2) : Extract the second derivative with respect to y
        elif (x_derivative[index], y_derivative[index]) == (0, 2):
            drho_dy = list_rho_eq[2]
            dtheta_dy = list_theta_eq[2]
            d2rho_dy2 = list_rho_eq[4]
            d2theta_dy2 = list_theta_eq[4]
            dZ_drho = output_polynomials[(1 - shift_0_derivative) * len(n) + index]
            dZ_dtheta = output_polynomials[(2 - shift_0_derivative) * len(n) + index]
            d2Z_drho2 = output_polynomials[(3 - shift_0_derivative) * len(n) + index]
            d2Z_dtheta2 = output_polynomials[(4 - shift_0_derivative) * len(n) + index]
            d2Z_drhodtheta = output_polynomials[(5 - shift_0_derivative) * len(n) + index]
            d2Z_dy2 = (d2Z_drho2 * drho_dy[valid_mask] + d2Z_drhodtheta * dtheta_dy[valid_mask]) * drho_dy[valid_mask] + \
                      dZ_drho * d2rho_dy2[valid_mask] + \
                      (d2Z_dtheta2 * dtheta_dy[valid_mask] + d2Z_drhodtheta * drho_dy[valid_mask]) * dtheta_dy[valid_mask] + \
                      dZ_dtheta * d2theta_dy2[valid_mask]
            output.append(d2Z_dy2)

        # Case (1, 1) : Extract the mixed second derivative
        elif (x_derivative[index], y_derivative[index]) == (1, 1):
            drho_dx = list_rho_eq[1]
            dtheta_dx = list_theta_eq[1]
            drho_dy = list_rho_eq[2]
            dtheta_dy = list_theta_eq[2]
            d2rho_dxdy = list_rho_eq[5]
            d2theta_dxdy = list_theta_eq[5]
            dZ_drho = output_polynomials[(1 - shift_0_derivative) * len(n) + index]
            dZ_dtheta = output_polynomials[(2 - shift_0_derivative) * len(n) + index]
            d2Z_drho2 = output_polynomials[(3 - shift_0_derivative) * len(n) + index]
            d2Z_dtheta2 = output_polynomials[(4 - shift_0_derivative) * len(n) + index]
            d2Z_drhodtheta = output_polynomials[(5 - shift_0_derivative) * len(n) + index]
            d2Z_dxdy = (d2Z_drho2 * drho_dx[valid_mask] + d2Z_drhodtheta * dtheta_dx[valid_mask]) * drho_dy[valid_mask] + \
                       dZ_drho * d2rho_dxdy[valid_mask] + \
                       (d2Z_dtheta2 * dtheta_dx[valid_mask] + d2Z_drhodtheta * drho_dx[valid_mask]) * dtheta_dy[valid_mask] + \
                       dZ_dtheta * d2theta_dxdy[valid_mask]
            output.append(d2Z_dxdy)

    # Reinsert the values into the original shape and set the default value for invalid points
    for index in range(len(output)):
        full_output = numpy.full(original_shape, default, dtype=float_type)
        full_output[valid_mask] = output[index]
        output[index] = full_output

    return output