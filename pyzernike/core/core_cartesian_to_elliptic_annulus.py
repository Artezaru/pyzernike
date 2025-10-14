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
from typing import Tuple, Sequence, List
import sympy

def core_cartesian_to_elliptic_annulus(
    x: numpy.ndarray,
    y: numpy.ndarray,
    Rx: float,
    Ry: float,
    x0: float,
    y0: float,
    alpha: float,
    h: float, 
    x_derivative: Sequence[int],
    y_derivative: Sequence[int],
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
        The x coordinates in Cartesian system with shape (...,).

    y : numpy.ndarray
        The y coordinates in Cartesian system with shape (...,).

    Rx : float
        The length of the semi-axis of the ellipse along x axis. Must be strictly positive.

    Ry : float
        The length of the semi-axis of the ellipse along y axis. Must be strictly positive.

    x0 : float
        The x coordinate of the center of the ellipse. Can be any real number.

    y0 : float
        The y coordinate of the center of the ellipse. Can be any real number.
 
    alpha : float
        The rotation angle of the ellipse in radians. Can be any real number.
    
    h : float
        The ratio of the inner semi-axis to the outer semi-axis. Must be in the range [0, 1).

    x_derivative : Sequence[int]
        The derivative order with respect to x to compute. Must be a sequence of non-negative integers.

    y_derivative : Sequence[int]
        The derivative order with respect to y to compute. Must be a sequence of non-negative integers of the same length as `x_derivative`.

    Returns
    -------
    Tuple[List[numpy.ndarray], List[numpy.ndarray]]

        The polar coordinates (:math:`\rho_{eq}, \theta_{eq}`) and their derivatives with respect to the Cartesian coordinates :math:`(x, y)`.
        ``output[0][i]`` is the derivative with respect to x of order ``x_derivative[i]`` and with respect to y of order ``y_derivative[i]`` of :math:`\rho_{eq}`.
        ``output[1][i]`` is the derivative with respect to x of order ``x_derivative[i]`` and with respect to y of order ``y_derivative[i]`` of :math:`\theta_{eq}`.

    Notes
    -----

    The derivatives for orders higher than 2 are computed using symbolic differentiation with sympy library (high computational cost).
    For orders 0, 1 and 2, the derivatives are computed using the analytical expressions derived from the chain rule.

    """
    # =================================================================================
    # Check if any derivative is lower/upper than 2 
    # =================================================================================
    derivative_total_order = [x_derivative[idx] + y_derivative[idx] for idx in range(len(x_derivative))]
    use_sympy = numpy.any(numpy.array(derivative_total_order) > 2)

    # =================================================================================
    # Fist computing the intermediate values and sympy expressions
    # =================================================================================

    # Prepare the sympy expression
    x_sympy = sympy.symbols('x')
    y_sympy = sympy.symbols('y')

    # Computing the X, Y arrays from x and y coordinates
    X = numpy.cos(alpha) * (x - x0) + numpy.sin(alpha) * (y - y0)
    X_sympy = sympy.cos(alpha) * (x_sympy - x0) + sympy.sin(alpha) * (y_sympy - y0) if use_sympy else None
    Y = - numpy.sin(alpha) * (x - x0) + numpy.cos(alpha) * (y - y0)
    Y_sympy = - sympy.sin(alpha) * (x_sympy - x0) + sympy.cos(alpha) * (y_sympy - y0) if use_sympy else None

    # Computing the derivative of X and Y along x and y 
    dX_dx = numpy.cos(alpha)
    dX_dy = numpy.sin(alpha)
    dY_dx = - numpy.sin(alpha)
    dY_dy = numpy.cos(alpha)

    # Compute the equivalent polar coordinates (With Angular in -pi to pi)
    r = numpy.sqrt((X / Rx) ** 2 + (Y / Ry) ** 2)
    r_sympy = sympy.sqrt((X_sympy / Rx) ** 2 + (Y_sympy / Ry) ** 2) if use_sympy else None
    theta = numpy.arctan2(Y / Ry, X / Rx)
    theta_sympy = sympy.atan2(Y_sympy / Ry, X_sympy / Rx) if use_sympy else None

    # Compute the equivalent rho values
    rho_eq = (r - h) / (1 - h)
    rho_eq_sympy = (r_sympy - h) / (1 - h) if use_sympy else None
    theta_eq = theta
    theta_eq_sympy = theta_sympy if use_sympy else None

    # =================================================================================
    # Now computing the derivatives
    # =================================================================================

    rho_eq_list = []
    theta_eq_list = []

    for idx in range(len(x_derivative)):
        dx_idx = x_derivative[idx]
        dy_idx = y_derivative[idx]

        # Case (0): dx = 0 and dy = 0
        if dx_idx == 0 and dy_idx == 0:
            rho_eq_idx = rho_eq.copy()
            theta_eq_idx = theta_eq.copy()

        # Case (1): dx + dy = 1
        elif dx_idx + dy_idx == 1:
            if dx_idx == 1:
                dX_dz = dX_dx
                dY_dz = dY_dx
            else: # dy_idx == 1
                dX_dz = dX_dy
                dY_dz = dY_dy
            rho_eq_idx = (1/(1-h)) * (1/r) * ((X*dX_dz)/(Rx**2) + (Y*dY_dz)/(Ry**2))
            theta_eq_idx = (1/(Rx*Ry)) * (1/(r**2)) * (dY_dz*X - dX_dz*Y)

        # Case (2): dx + dy = 2
        elif dx_idx + dy_idx == 2:
            if dx_idx == 2:
                dX_dz = dX_dx
                dY_dz = dY_dx
                dX_dw = dX_dx
                dY_dw = dY_dx
            elif dy_idx == 2:
                dX_dz = dX_dy
                dY_dz = dY_dy
                dX_dw = dX_dy
                dY_dw = dY_dy
            else: # dx_idx == 1 and dy_idx == 1
                dX_dz = dX_dx
                dY_dz = dY_dx
                dX_dw = dX_dy
                dY_dw = dY_dy
            rho_eq_idx = (1/(1-h)) * ((1/r) * ((dX_dw*dX_dz)/(Rx**2) + (dY_dw*dY_dz)/(Ry**2)) - (1/r**3) * ((X*dX_dw)/(Rx**2) + (Y*dY_dw)/(Ry**2)) * ((X*dX_dz)/(Rx**2) + (Y*dY_dz)/(Ry**2)))
            theta_eq_idx = (1/(Rx*Ry)) * ((1/(r**2)) * (dX_dw*dY_dz - dY_dw*dX_dz) - (2/(r**4)) * (X*dY_dz - Y*dX_dz) * ((X*dX_dw)/(Rx**2) + (Y*dY_dw)/(Ry**2)))

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
        rho_eq_list.append(rho_eq_idx)
        theta_eq_list.append(theta_eq_idx)

    # Returning the final lists
    return rho_eq_list, theta_eq_list
