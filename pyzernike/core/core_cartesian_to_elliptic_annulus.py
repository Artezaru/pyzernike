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
from typing import Tuple, List
import sympy

from .core_corresponding_signed_integer_type import core_corresponding_signed_integer_type

def core_cartesian_to_elliptic_annulus(
    x: numpy.ndarray,
    y: numpy.ndarray,
    Rx: numpy.floating,
    Ry: numpy.floating,
    x0: numpy.floating,
    y0: numpy.floating,
    alpha: numpy.floating,
    h: numpy.floating, 
    x_derivative: numpy.ndarray,
    y_derivative: numpy.ndarray,
    float_type: type[numpy.floating]
    ) -> Tuple[List[numpy.ndarray], List[numpy.ndarray]]:
    r"""
    Transform Cartesian coordinates :math:`(x, y)` to elliptic annulus domain polar coordinates :math:`(\rho_{eq}, \theta_{eq})`.

    .. warning::

        This method is a core function of ``pyzernike`` that is not designed to be use by the users directly.
        No test is done on the input parameters. Please use the high level functions.

    .. seealso::

        - :func:`pyzernike.cartesian_to_elliptic_annulus` to convert Cartesian coordinates to elliptic annulus domain polar coordinates.
        - :func:`pyzernike.xy_zernike_polynomial` to compute Zernike polynomials on the elliptic annulus domain.
        - The page :doc:`../../mathematical_description` in the documentation for the mathematical extension of the Zernike polynomials on the elliptic domain.

    Lets consider the extended elliptic annulus domain defined by the following parameters:

    .. figure:: ../../../../pyzernike/resources/elliptic_annulus_domain.png
        :width: 400px
        :align: center

        The parameters to define the extended domain of the Zernike polynomial.    

    The parameters are:

    - :math:`R_x` and :math:`R_y` are the lengths of the semi-axis of the ellipse.
    - :math:`x_0` and :math:`y_0` are the coordinates of the center of the ellipse.
    - :math:`\alpha` is the rotation angle of the ellipse in radians.
    - :math:`h=\frac{a}{R_x}=\frac{b}{R_y}` defining the inner boundary of the ellipse.

    The methods allow to compute the polar coordinates :math:`(\rho_{eq}, \theta_{eq})` and their derivatives with respect to the Cartesian coordinates :math:`(x, y)`.

    - ``x`` and ``y`` are expected to be numpy arrays of the same shape and same dtype.
    - ``x_derivative`` and ``y_derivative`` must be sequences of non-negative integers of the same length.

    The output is a tuple of two lists with lengths equal to the length of ``x_derivative`` and ``y_derivative``:

    - The first list contains the equivalent polar radius :math:`\rho_{eq}` and its derivatives with respect to the given orders.
    - The second list contains the equivalent polar angle :math:`\theta_{eq}` and its derivatives with respect to the given orders.

    Parameters
    ----------
    x : numpy.ndarray
        The x coordinates in Cartesian system with shape (...,). Must be array and floating point dtype corresponding to `float_type`.

    y : numpy.ndarray
        The y coordinates in Cartesian system with shape (...,). Must be array and floating point dtype corresponding to `float_type`.

    Rx : numpy.floating
        The length of the semi-axis of the ellipse along x axis. Must be strictly positive and floating point type corresponding to `float_type`.

    Ry : numpy.floating
        The length of the semi-axis of the ellipse along y axis. Must be strictly positive and floating point type corresponding to `float_type`.

    x0 : numpy.floating
        The x coordinate of the center of the ellipse. Can be any real number as floating point type corresponding to `float_type`.

    y0 : numpy.floating
        The y coordinate of the center of the ellipse. Can be any real number as floating point type corresponding to `float_type`.
 
    alpha : numpy.floating
        The rotation angle of the ellipse in radians. Can be any real number as floating point type corresponding to `float_type`.
    
    h : numpy.floating
        The ratio of the inner semi-axis to the outer semi-axis. Must be in the range [0, 1) as floating point type corresponding to `float_type`.

    x_derivative : numpy.ndarray
        The derivative order with respect to x to compute. Must be a sequence of non-negative integers with type compatible to `float_type`.

    y_derivative : numpy.ndarray
        The derivative order with respect to y to compute. Must be a sequence of non-negative integers of the same length as `x_derivative` with type compatible to `float_type`.

    Returns
    -------
    Tuple[List[numpy.ndarray], List[numpy.ndarray]]

        The polar coordinates (:math:`\rho_{eq}, \theta_{eq}`) and their derivatives with respect to the Cartesian coordinates :math:`(x, y)` as two lists of numpy arrays of floating point type corresponding to `float_type`.
        ``output[0][i]`` is the derivative with respect to x of order ``x_derivative[i]`` and with respect to y of order ``y_derivative[i]`` of :math:`\rho_{eq}`.
        ``output[1][i]`` is the derivative with respect to x of order ``x_derivative[i]`` and with respect to y of order ``y_derivative[i]`` of :math:`\theta_{eq}`.

    Notes
    -----

    The derivatives for orders higher than 2 are computed using symbolic differentiation with sympy library (high computational cost).
    For orders 0, 1 and 2, the derivatives are computed using the analytical expressions derived from the chain rule.

    """
    # Get the corresponding integer types
    int_type = core_corresponding_signed_integer_type(float_type)

    # Fast assertions on the inputs
    assert issubclass(float_type, numpy.floating), "[pyzernike-core] float_type must be a numpy floating point type."
    assert isinstance(x, numpy.ndarray) and numpy.issubdtype(x.dtype, float_type), "[pyzernike-core] x must be a numpy array of floating point values of type compatible with float_type."
    assert isinstance(y, numpy.ndarray) and numpy.issubdtype(y.dtype, float_type), "[pyzernike-core] y must be a numpy array of floating point values of type compatible with float_type."
    assert isinstance(Rx, float_type), "[pyzernike-core] Rx must be a floating point value of type compatible with float_type."
    assert Rx > 0, "[pyzernike-core] Rx must be strictly positive."
    assert isinstance(Ry, float_type), "[pyzernike-core] Ry must be a floating point value of type compatible with float_type."
    assert Ry > 0, "[pyzernike-core] Ry must be strictly positive."
    assert isinstance(x0, float_type), "[pyzernike-core] x0 must be a floating point value of type compatible with float_type."
    assert isinstance(y0, float_type), "[pyzernike-core] y0 must be a floating point value of type compatible with float_type."
    assert isinstance(alpha, float_type), "[pyzernike-core] alpha must be a floating point value of type compatible with float_type."
    assert isinstance(h, float_type), "[pyzernike-core] h must be a floating point value of type compatible with float_type."
    assert 0 <= h < 1, "[pyzernike-core] h must be in the range [0, 1)."
    assert isinstance(x_derivative, numpy.ndarray) and x_derivative.ndim == 1 and numpy.issubdtype(x_derivative.dtype, int_type), "[pyzernike-core] x_derivative must be a 1D numpy array of integers of type compatible with float_type."
    assert isinstance(y_derivative, numpy.ndarray) and y_derivative.ndim == 1 and numpy.issubdtype(y_derivative.dtype, int_type), "[pyzernike-core] y_derivative must be a 1D numpy array of integers of type compatible with float_type."
    assert x_derivative.size == y_derivative.size, "[pyzernike-core] x_derivative and y_derivative must have the same size."

    # =================================================================================
    # Check if any derivative is lower/upper than 2 
    # =================================================================================
    derivative_total_order: List[numpy.integer] = [x_derivative[idx] + y_derivative[idx] for idx in range(len(x_derivative))]
    use_sympy: bool = numpy.any(numpy.array(derivative_total_order) > 2)

    # =================================================================================
    # Fist computing the intermediate values and sympy expressions
    # =================================================================================

    # Prepare the sympy expression
    x_sympy = sympy.symbols('x')
    y_sympy = sympy.symbols('y')

    # Computing the X, Y arrays from x and y coordinates
    X: numpy.ndarray = numpy.cos(alpha, dtype=float_type) * (x - x0) + numpy.sin(alpha, dtype=float_type) * (y - y0)
    X_sympy = sympy.cos(alpha) * (x_sympy - x0) + sympy.sin(alpha) * (y_sympy - y0) if use_sympy else None
    Y: numpy.ndarray = - numpy.sin(alpha, dtype=float_type) * (x - x0) + numpy.cos(alpha, dtype=float_type) * (y - y0)
    Y_sympy = - sympy.sin(alpha) * (x_sympy - x0) + sympy.cos(alpha) * (y_sympy - y0) if use_sympy else None

    # Computing the derivative of X and Y along x and y 
    dX_dx: numpy.floating = numpy.cos(alpha, dtype=float_type)
    dX_dy: numpy.floating = numpy.sin(alpha, dtype=float_type)
    dY_dx: numpy.floating = - numpy.sin(alpha, dtype=float_type)
    dY_dy: numpy.floating = numpy.cos(alpha, dtype=float_type)

    # Compute the equivalent polar coordinates (With Angular in -pi to pi)
    r: numpy.floating = numpy.sqrt((X / Rx) ** 2 + (Y / Ry) ** 2, dtype=float_type)
    r_sympy = sympy.sqrt((X_sympy / Rx) ** 2 + (Y_sympy / Ry) ** 2) if use_sympy else None
    theta: numpy.floating = numpy.arctan2(Y / Ry, X / Rx, dtype=float_type)
    theta_sympy = sympy.atan2(Y_sympy / Ry, X_sympy / Rx) if use_sympy else None

    # Compute the equivalent rho values
    rho_eq: numpy.ndarray = (r - h) / (1 - h)
    rho_eq_sympy = (r_sympy - h) / (1 - h) if use_sympy else None
    theta_eq: numpy.ndarray = theta
    theta_eq_sympy = theta_sympy if use_sympy else None

    # =================================================================================
    # Now computing the derivatives
    # =================================================================================

    rho_eq_list: list[numpy.ndarray] = []
    theta_eq_list: list[numpy.ndarray] = []

    for idx in range(len(x_derivative)):
        dx_idx: numpy.integer = x_derivative[idx]
        dy_idx: numpy.integer = y_derivative[idx]

        # Case (0): dx = 0 and dy = 0
        if dx_idx == 0 and dy_idx == 0:
            rho_eq_idx: numpy.ndarray = rho_eq.copy()
            theta_eq_idx: numpy.ndarray = theta_eq.copy()

        # Case (1): dx + dy = 1
        elif dx_idx + dy_idx == 1:
            if dx_idx == 1:
                dX_dz: numpy.floating = dX_dx
                dY_dz: numpy.floating = dY_dx
            else: # dy_idx == 1
                dX_dz: numpy.floating = dX_dy
                dY_dz: numpy.floating = dY_dy
            rho_eq_idx: numpy.ndarray = (1/(1-h)) * (1/r) * ((X*dX_dz)/(Rx**2) + (Y*dY_dz)/(Ry**2))
            theta_eq_idx: numpy.ndarray = (1/(Rx*Ry)) * (1/(r**2)) * (dY_dz*X - dX_dz*Y)

        # Case (2): dx + dy = 2
        elif dx_idx + dy_idx == 2:
            if dx_idx == 2:
                dX_dz: numpy.floating = dX_dx
                dY_dz: numpy.floating = dY_dx
                dX_dw: numpy.floating = dX_dx
                dY_dw: numpy.floating = dY_dx
            elif dy_idx == 2:
                dX_dz: numpy.floating = dX_dy
                dY_dz: numpy.floating = dY_dy
                dX_dw: numpy.floating = dX_dy
                dY_dw: numpy.floating = dY_dy
            else: # dx_idx == 1 and dy_idx == 1
                dX_dz: numpy.floating = dX_dx
                dY_dz: numpy.floating = dY_dx
                dX_dw: numpy.floating = dX_dy
                dY_dw: numpy.floating = dY_dy
            rho_eq_idx: numpy.ndarray = (1/(1-h)) * ((1/r) * ((dX_dw*dX_dz)/(Rx**2) + (dY_dw*dY_dz)/(Ry**2)) - (1/r**3) * ((X*dX_dw)/(Rx**2) + (Y*dY_dw)/(Ry**2)) * ((X*dX_dz)/(Rx**2) + (Y*dY_dz)/(Ry**2)))
            theta_eq_idx: numpy.ndarray = (1/(Rx*Ry)) * ((1/(r**2)) * (dX_dw*dY_dz - dY_dw*dX_dz) - (2/(r**4)) * (X*dY_dz - Y*dX_dz) * ((X*dX_dw)/(Rx**2) + (Y*dY_dw)/(Ry**2)))

        # Case (3) and more: dx + dy >= 3
        elif use_sympy:
            # Using sympy to compute the derivatives
            rho_eq_expression = sympy.diff(rho_eq_sympy, x_sympy, dx_idx, y_sympy, dy_idx)
            rho_eq_expression = sympy.simplify(rho_eq_expression)
            rho_eq_func = sympy.lambdify((x_sympy, y_sympy), rho_eq_expression, modules='numpy')
            rho_eq_idx = rho_eq_func(x, y)
            theta_eq_expression = sympy.diff(theta_eq_sympy, x_sympy, dx_idx, y_sympy, dy_idx)
            theta_eq_expression = sympy.simplify(theta_eq_expression)
            theta_eq_func = sympy.lambdify((x_sympy, y_sympy), theta_eq_expression, modules='numpy')
            theta_eq_idx = theta_eq_func(x, y)

        else:
            raise ValueError("Derivative order too high and sympy is not enabled.")

        # Append the results to the lists
        assert rho_eq_idx.shape == x.shape, "[pyzernike-core] Internal error: rho_eq derivative shape mismatch."
        assert theta_eq_idx.shape == x.shape, "[pyzernike-core] Internal error: theta_eq derivative shape mismatch."
        assert rho_eq_idx.dtype == float_type, "[pyzernike-core] Internal error: rho_eq derivative dtype mismatch."
        assert theta_eq_idx.dtype == float_type, "[pyzernike-core] Internal error: theta_eq derivative dtype mismatch."
        
        rho_eq_list.append(rho_eq_idx)
        theta_eq_list.append(theta_eq_idx)

    # Returning the final lists
    return rho_eq_list, theta_eq_list
