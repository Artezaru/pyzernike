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

from .xy_zernike_polynomial import xy_zernike_polynomial
from .zernike_index_to_order import zernike_index_to_order
from .core.core_corresponding_signed_integer_type import core_corresponding_signed_integer_type

def xy_zernike_polynomial_up_to_order(
    x: numpy.ndarray,
    y: numpy.ndarray,
    order: Integral,
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
    Computes all the Zernike polynomial :math:`Z_{n}^{m}(\rho_{eq}, \theta_{eq})` for given cartesian coordinates :math:`(x, y)` on an elliptic annulus domain up to a given order.

    .. seealso::

        - :func:`pyzernike.xy_zernike_polynomial` for computing a set of Zernike polynomials for given orders and azimuthal frequencies.
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
    
    This function allows to compute Zernike polynomials at once for different sets of derivative orders given as sequences,
    which can be more efficient than calling the function multiple times for each set of derivative orders.

    - The parameters ``x`` and ``y`` must be numpy arrays of the same shape.
    - The parameters ``x_derivative`` and ``y_derivative`` must be sequences of integers with the same length.

    The :math:`x` and :math:`y` values are the same for all the polynomials.
    The output ``output[k][j]`` is the Zernike polynomial of order ``n[j]`` and azimuthal frequency ``m[j]`` (OSA/ANSI ordering) with same shape as ``x`` and for the radial derivative of order ``x_derivative[k]`` and the angular derivative of order ``y_derivative[k]``.

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

    order : int
        The maximum order of the Zernike polynomials to compute. It must be a positive integer.

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
        If x_derivative or y_derivative (if not None) are not sequences of integers.

    ValueError
        If the x and y do not have the same shape.
        If the lengths of x_derivative and y_derivative (if not None) are not the same.
        If Rx or Ry are not strictly positive.
        If h is not in the range [0, 1[.
        If the derivative orders are higher than 2 or mixed (ex: (1, 1)).

    Examples
    --------
    Compute all the Zernike polynomials up to order 3 for a cartesian grid of points in the domain defined by a circle with radius sqrt(2) centered at (0, 0):

    .. code-block:: python

        import numpy
        from pyzernike import xy_zernike_polynomial_up_to_order, zernike_index_to_order

        # Create a grid of points
        x = numpy.linspace(-1, 1, 100)
        y = numpy.linspace(-1, 1, 100)
        x, y = numpy.meshgrid(x, y)  # Create a 2D grid

        # Compute the Zernike polynomials up to order 3
        result = xy_zernike_polynomial_up_to_order(x, y, order=3, Rx=numpy.sqrt(2), Ry=numpy.sqrt(2), x0=0, y0=0)
        polynomials = result[0]  # Get the first set of polynomials (for x_derivative=0, y_derivative=0)

        # Extract the values: 
        indices = list(range(len(polynomials)))
        n, m = zernike_index_to_order(indices)  # Get the orders and azimuthal frequencies from the indices

        for i, (n_i, m_i) in enumerate(zip(n, m)):
            print(f"Zernike polynomial Z_{n_i}^{m_i} for the given x and y values is: {polynomials[i]}")

    To compute the polynomials and their first derivatives with respect to x:

    .. code-block:: python

        import numpy
        from pyzernike import xy_zernike_polynomial_up_to_order, zernike_index_to_order

        # Create a grid of points
        x = numpy.linspace(-1, 1, 100)
        y = numpy.linspace(-1, 1, 100)
        x, y = numpy.meshgrid(x, y)  # Create a 2D grid

        # Compute the Zernike polynomials up to order 3 with x derivatives
        result = xy_zernike_polynomial_up_to_order(x, y, order=3, x_derivative=[0, 1], Rx=numpy.sqrt(2), Ry=numpy.sqrt(2), x0=0, y0=0)
        polynomials = result[0]  # Get the first set of polynomials (for x_derivative=0, y_derivative=0)
        derivatives_x = result[1]  # Get the first set of derivatives (for x_derivative=1, y_derivative=0)

    The output will contain the Zernike polynomials and their derivatives for the specified orders and azimuthal frequencies.
    
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
    if not isinstance(order, Integral) or order < 0:
        raise TypeError("order must be a positive integer.")
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
    if x_derivative is None and y_derivative is None:
        x_derivative = numpy.array([0], dtype=int_type)
        y_derivative = numpy.array([0], dtype=int_type)
    elif x_derivative is None and y_derivative is not None:
        x_derivative = numpy.zeros_like(y_derivative, dtype=int_type)
        y_derivative = numpy.asarray(y_derivative, dtype=int_type)
    elif x_derivative is not None and y_derivative is None:
        y_derivative = numpy.zeros_like(x_derivative, dtype=int_type)
        x_derivative = numpy.asarray(x_derivative, dtype=int_type)
    else:
        x_derivative = numpy.asarray(x_derivative, dtype=int_type)
        y_derivative = numpy.asarray(y_derivative, dtype=int_type)
    
    # Check lengths
    if not x_derivative.ndim == 1:
        raise TypeError("x_derivative must be a sequence or a 1D array of integers.")
    if not y_derivative.ndim == 1:
        raise TypeError("y_derivative must be a sequence or a 1D array of integers.")
    if not (x_derivative.size == y_derivative.size):
        raise ValueError("x_derivative (if given) and y_derivative (if given) must have the same length.")

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

    # Create the [n,m,...] lists for all the Zernike polynomials up to the given order
    N_polynomials = (order + 1) * (order + 2) // 2
    N_derivatives = len(x_derivative)
    n, m = zernike_index_to_order(list(range(N_polynomials)))

    # Extend n and m to match the length of x_derivative and y_derivative
    n = n * len(x_derivative)
    m = m * len(x_derivative)
    x_derivative = [dr for dr in x_derivative for _ in range(N_polynomials)]
    y_derivative = [dt for dt in y_derivative for _ in range(N_polynomials)]

    # Compute the Zernike polynomials using the core_polynomial function
    output = xy_zernike_polynomial(
        x=x,
        y=y,
        n=n,
        m=m,
        Rx=Rx,
        Ry=Ry,
        x0=x0,
        y0=y0,
        alpha=alpha,
        h=h,
        x_derivative=x_derivative,
        y_derivative=y_derivative,
        default=default,
        precompute=precompute,
    ) # List[N_polys * len(rho_derivative) of numpy.ndarray with shape of valid rho]

    # Reshape the output to a list of lists of shape (len(rho_derivative), N_polynomials)
    output = [output[i * N_polynomials:(i + 1) * N_polynomials] for i in range(N_derivatives)]

    return output
