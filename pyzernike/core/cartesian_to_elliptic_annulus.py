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

from typing import List, Tuple

import numpy
import sympy


X_SYMBOL = sympy.Symbol("x")
Y_SYMBOL = sympy.Symbol("y")


def _first_order_derivatives(
    X: numpy.ndarray,
    Y: numpy.ndarray,
    r: numpy.ndarray,
    Rx: float,
    Ry: float,
    h: float,
    dX_dz: float,
    dY_dz: float,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    r"""
    First order derivatives of :math:`(\rho_{eq}, \theta_{eq})` with respect to one Cartesian coordinate :math:`z \in \{x, y\}`.

    .. math::

        \frac{\partial \rho_{eq}}{\partial z} = \frac{1}{1-h} \frac{1}{r} \left( \frac{X X_z}{R_x^2} + \frac{Y Y_z}{R_y^2} \right)
        \quad \text{and} \quad
        \frac{\partial \theta_{eq}}{\partial z} = \frac{1}{R_x R_y} \frac{1}{r^2} \left( Y_z X - X_z Y \right)

    where :math:`X_z = \frac{\partial X}{\partial z}` and :math:`Y_z = \frac{\partial Y}{\partial z}` are constants.

    Parameters
    ----------
    X, Y : numpy.ndarray
        The coordinates in the frame of the ellipse.

    r : numpy.ndarray
        The normalized elliptic radius :math:`r = \sqrt{(X/R_x)^2 + (Y/R_y)^2}`.

    Rx, Ry : float
        The semi-axes of the ellipse.

    h : float
        The ratio of the inner to the outer semi-axes.

    dX_dz, dY_dz : float
        The derivatives :math:`X_z` and :math:`Y_z`.

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray]
        The derivatives :math:`\frac{\partial \rho_{eq}}{\partial z}` and :math:`\frac{\partial \theta_{eq}}{\partial z}`.

    """
    rho_eq_derivative = (1/(1-h)) * (1/r) * ((X*dX_dz)/(Rx**2) + (Y*dY_dz)/(Ry**2))
    theta_eq_derivative = (1/(Rx*Ry)) * (1/(r**2)) * (dY_dz*X - dX_dz*Y)
    return rho_eq_derivative, theta_eq_derivative


def _second_order_derivatives(
    X: numpy.ndarray,
    Y: numpy.ndarray,
    r: numpy.ndarray,
    Rx: float,
    Ry: float,
    h: float,
    dX_dz: float,
    dY_dz: float,
    dX_dw: float,
    dY_dw: float,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    r"""
    Second order derivatives of :math:`(\rho_{eq}, \theta_{eq})` with respect to two Cartesian coordinates :math:`z, w \in \{x, y\}`.

    .. math::

        \frac{\partial^2 \rho_{eq}}{\partial z \partial w} = \frac{1}{1-h} \left[ \frac{1}{r} \left( \frac{X_w X_z}{R_x^2} + \frac{Y_w Y_z}{R_y^2} \right) - \frac{1}{r^3} \left( \frac{X X_w}{R_x^2} + \frac{Y Y_w}{R_y^2} \right) \left( \frac{X X_z}{R_x^2} + \frac{Y Y_z}{R_y^2} \right) \right]

    .. math::

        \frac{\partial^2 \theta_{eq}}{\partial z \partial w} = \frac{1}{R_x R_y} \left[ \frac{1}{r^2} \left( X_w Y_z - Y_w X_z \right) - \frac{2}{r^4} \left( X Y_z - Y X_z \right) \left( \frac{X X_w}{R_x^2} + \frac{Y Y_w}{R_y^2} \right) \right]

    Parameters
    ----------
    X, Y : numpy.ndarray
        The coordinates in the frame of the ellipse.

    r : numpy.ndarray
        The normalized elliptic radius :math:`r = \sqrt{(X/R_x)^2 + (Y/R_y)^2}`.

    Rx, Ry : float
        The semi-axes of the ellipse.

    h : float
        The ratio of the inner to the outer semi-axes.

    dX_dz, dY_dz : float
        The derivatives :math:`X_z` and :math:`Y_z`.

    dX_dw, dY_dw : float
        The derivatives :math:`X_w` and :math:`Y_w`.

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray]
        The derivatives :math:`\frac{\partial^2 \rho_{eq}}{\partial z \partial w}` and :math:`\frac{\partial^2 \theta_{eq}}{\partial z \partial w}`.

    """
    rho_eq_derivative = (1/(1-h)) * ((1/r) * ((dX_dw*dX_dz)/(Rx**2) + (dY_dw*dY_dz)/(Ry**2)) - (1/r**3) * ((X*dX_dw)/(Rx**2) + (Y*dY_dw)/(Ry**2)) * ((X*dX_dz)/(Rx**2) + (Y*dY_dz)/(Ry**2)))
    theta_eq_derivative = (1/(Rx*Ry)) * ((1/(r**2)) * (dX_dw*dY_dz - dY_dw*dX_dz) - (2/(r**4)) * (X*dY_dz - Y*dX_dz) * ((X*dX_dw)/(Rx**2) + (Y*dY_dw)/(Ry**2)))
    return rho_eq_derivative, theta_eq_derivative


def _symbolic_elliptic_annulus(
    Rx: float,
    Ry: float,
    x0: float,
    y0: float,
    alpha: float,
    h: float,
) -> Tuple[sympy.Expr, sympy.Expr]:
    r"""
    Build the symbolic expressions of :math:`\rho_{eq}(x, y)` and :math:`\theta_{eq}(x, y)` in the symbols ``x`` and ``y``.

    Parameters
    ----------
    Rx, Ry : float
        The semi-axes of the ellipse.

    x0, y0 : float
        The center of the ellipse.

    alpha : float
        The rotation angle of the ellipse in radians.

    h : float
        The ratio of the inner to the outer semi-axes.

    Returns
    -------
    Tuple[sympy.Expr, sympy.Expr]
        The expressions of :math:`\rho_{eq}` and :math:`\theta_{eq}`.

    """
    X_sympy = sympy.cos(alpha) * (X_SYMBOL - x0) + sympy.sin(alpha) * (Y_SYMBOL - y0)
    Y_sympy = - sympy.sin(alpha) * (X_SYMBOL - x0) + sympy.cos(alpha) * (Y_SYMBOL - y0)
    r_sympy = sympy.sqrt((X_sympy / Rx) ** 2 + (Y_sympy / Ry) ** 2)
    theta_sympy = sympy.atan2(Y_sympy / Ry, X_sympy / Rx)
    rho_eq_sympy = (r_sympy - h) / (1 - h)
    return rho_eq_sympy, theta_sympy


def _symbolic_derivatives(
    rho_eq_sympy: sympy.Expr,
    theta_eq_sympy: sympy.Expr,
    x: numpy.ndarray,
    y: numpy.ndarray,
    x_derivative: int,
    y_derivative: int,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    r"""
    Evaluate the derivatives of :math:`(\rho_{eq}, \theta_{eq})` of any order by symbolic differentiation.

    The expressions are differentiated with ``sympy.diff``, simplified, and evaluated on ``(x, y)``
    with ``sympy.lambdify``.

    Parameters
    ----------
    rho_eq_sympy, theta_eq_sympy : sympy.Expr
        The expressions of :math:`\rho_{eq}` and :math:`\theta_{eq}` (see :func:`_symbolic_elliptic_annulus`).

    x, y : numpy.ndarray
        The Cartesian coordinates.

    x_derivative, y_derivative : int
        The orders of the derivatives with respect to :math:`x` and :math:`y`.

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray]
        The derivatives of :math:`\rho_{eq}` and :math:`\theta_{eq}` evaluated on ``(x, y)``.

    """
    rho_eq_expression = sympy.simplify(sympy.diff(rho_eq_sympy, X_SYMBOL, x_derivative, Y_SYMBOL, y_derivative))
    rho_eq_derivative = sympy.lambdify((X_SYMBOL, Y_SYMBOL), rho_eq_expression, modules='numpy')(x, y)
    theta_eq_expression = sympy.simplify(sympy.diff(theta_eq_sympy, X_SYMBOL, x_derivative, Y_SYMBOL, y_derivative))
    theta_eq_derivative = sympy.lambdify((X_SYMBOL, Y_SYMBOL), theta_eq_expression, modules='numpy')(x, y)
    return rho_eq_derivative, theta_eq_derivative


def _cartesian_to_elliptic_annulus(
    x: numpy.ndarray,
    y: numpy.ndarray,
    Rx: float,
    Ry: float,
    x0: float,
    y0: float,
    alpha: float,
    h: float,
    x_derivative: List[int],
    y_derivative: List[int],
) -> Tuple[List[numpy.ndarray], List[numpy.ndarray]]:
    r"""
    Transform Cartesian coordinates :math:`(x, y)` into the equivalent polar coordinates
    :math:`(\rho_{eq}, \theta_{eq})` of an elliptic annulus, and compute their derivatives with
    respect to :math:`x` and :math:`y`.

    .. figure:: ../../../../pyzernike/resources/elliptic_annulus_domain.png
        :width: 400px
        :align: center

        The parameters of the elliptic annulus domain.

    The elliptic annulus is defined by its semi-axes :math:`R_x` and :math:`R_y`, its center
    :math:`(x_0, y_0)`, its rotation angle :math:`\alpha` and the ratio :math:`h = \frac{a}{R_x} = \frac{b}{R_y}`
    of the inner to the outer semi-axes. The coordinates are first expressed in the frame of the ellipse:

    .. math::

        X = \cos(\alpha) (x - x_0) + \sin(\alpha) (y - y_0)
        \quad \text{and} \quad
        Y = -\sin(\alpha) (x - x_0) + \cos(\alpha) (y - y_0)

    then mapped onto the unit annulus [1]_:

    .. math::

        r = \sqrt{\left(\frac{X}{R_x}\right)^2 + \left(\frac{Y}{R_y}\right)^2},
        \quad
        \rho_{eq} = \frac{r - h}{1 - h}
        \quad \text{and} \quad
        \theta_{eq} = \operatorname{atan2}\left(\frac{Y}{R_y}, \frac{X}{R_x}\right) \in [-\pi, \pi]

    The derivatives of total order 1 and 2 are computed with the analytical expressions obtained by
    the chain rule (see :func:`_first_order_derivatives` and :func:`_second_order_derivatives`).
    The derivatives of total order 3 and more are computed by symbolic differentiation with ``sympy``
    (see :func:`_symbolic_derivatives`), which has a high computational cost.

    Parameters
    ----------
    x : numpy.ndarray
        The x Cartesian coordinates with shape ``(...,)`` and floating point dtype.

    y : numpy.ndarray
        The y Cartesian coordinates with same shape and dtype as ``x``.

    Rx : float
        The semi-axis of the ellipse along its x axis. Must be strictly positive.

    Ry : float
        The semi-axis of the ellipse along its y axis. Must be strictly positive.

    x0 : float
        The x coordinate of the center of the ellipse.

    y0 : float
        The y coordinate of the center of the ellipse.

    alpha : float
        The rotation angle of the ellipse in radians.

    h : float
        The ratio of the inner to the outer semi-axes, in :math:`[0, 1)`.

    x_derivative : List[int]
        The orders of the derivatives with respect to x.

    y_derivative : List[int]
        The orders of the derivatives with respect to y. Must have the same length as ``x_derivative``.

    Returns
    -------
    Tuple[List[numpy.ndarray], List[numpy.ndarray]]
        The lists of :math:`\rho_{eq}` and :math:`\theta_{eq}` derivatives. ``output[0][i]`` (resp.
        ``output[1][i]``) is the derivative of :math:`\rho_{eq}` (resp. :math:`\theta_{eq}`) of order
        ``x_derivative[i]`` with respect to x and ``y_derivative[i]`` with respect to y, with the shape of ``x``.

    References
    ----------
    .. [1] R. Navarro et al., "Generalization of Zernike polynomials for regular portions of circles
       and ellipses," Optics Express 22(18), 21263-21279 (2014). https://doi.org/10.1364/OE.22.021263

    """
    dtype = x.dtype
    use_sympy = any(dx + dy > 2 for dx, dy in zip(x_derivative, y_derivative))

    X = numpy.cos(alpha, dtype=dtype) * (x - x0) + numpy.sin(alpha, dtype=dtype) * (y - y0)
    Y = - numpy.sin(alpha, dtype=dtype) * (x - x0) + numpy.cos(alpha, dtype=dtype) * (y - y0)

    dX_dx = numpy.cos(alpha, dtype=dtype)
    dX_dy = numpy.sin(alpha, dtype=dtype)
    dY_dx = - numpy.sin(alpha, dtype=dtype)
    dY_dy = numpy.cos(alpha, dtype=dtype)

    r = numpy.sqrt((X / Rx) ** 2 + (Y / Ry) ** 2, dtype=dtype)
    theta = numpy.arctan2(Y / Ry, X / Rx, dtype=dtype)

    rho_eq = (r - h) / (1 - h)
    theta_eq = theta

    if use_sympy:
        rho_eq_sympy, theta_eq_sympy = _symbolic_elliptic_annulus(Rx, Ry, x0, y0, alpha, h)

    rho_eq_list: List[numpy.ndarray] = []
    theta_eq_list: List[numpy.ndarray] = []

    for dx, dy in zip(x_derivative, y_derivative):

        if dx == 0 and dy == 0:
            rho_eq_idx = rho_eq.copy()
            theta_eq_idx = theta_eq.copy()

        elif dx + dy == 1:
            dX_dz, dY_dz = (dX_dx, dY_dx) if dx == 1 else (dX_dy, dY_dy)
            rho_eq_idx, theta_eq_idx = _first_order_derivatives(X, Y, r, Rx, Ry, h, dX_dz, dY_dz)

        elif dx + dy == 2:
            if dx == 2:
                dX_dz, dY_dz, dX_dw, dY_dw = dX_dx, dY_dx, dX_dx, dY_dx
            elif dy == 2:
                dX_dz, dY_dz, dX_dw, dY_dw = dX_dy, dY_dy, dX_dy, dY_dy
            else:
                dX_dz, dY_dz, dX_dw, dY_dw = dX_dx, dY_dx, dX_dy, dY_dy
            rho_eq_idx, theta_eq_idx = _second_order_derivatives(X, Y, r, Rx, Ry, h, dX_dz, dY_dz, dX_dw, dY_dw)

        else:
            rho_eq_idx, theta_eq_idx = _symbolic_derivatives(rho_eq_sympy, theta_eq_sympy, x, y, dx, dy)

        rho_eq_list.append(rho_eq_idx)
        theta_eq_list.append(theta_eq_idx)

    return rho_eq_list, theta_eq_list