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

import math
from typing import List, Tuple

import numpy
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox

from ..core.core_cartesian_to_elliptic_annulus import _core_cartesian_to_elliptic_annulus
from .core_display import N_SAMPLES, _parse_integer, _parse_positive_float
from ..core.core_xy_polynomial import _core_xy_polynomials


def _xy_display_grid(
    Rx: float,
    Ry: float,
    x0: float,
    y0: float,
    alpha: float,
    h: float,
    resolution: int,
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    r"""
    Build a Cartesian display grid covering the elliptic annulus, and the equivalent polar coordinates of its points.

    The grid spans the bounding box of the rotated ellipse, enlarged by 5% on each side. The half-widths of
    the bounding box are:

    .. math::

        w_x = \sqrt{(R_x \cos\alpha)^2 + (R_y \sin\alpha)^2}
        \quad \text{and} \quad
        w_y = \sqrt{(R_x \sin\alpha)^2 + (R_y \cos\alpha)^2}

    Parameters
    ----------
    Rx : float
        The semi-axis :math:`R_x` of the ellipse along its x axis.

    Ry : float
        The semi-axis :math:`R_y` of the ellipse along its y axis.

    x0 : float
        The x coordinate :math:`x_0` of the center of the ellipse.

    y0 : float
        The y coordinate :math:`y_0` of the center of the ellipse.

    alpha : float
        The rotation angle :math:`\alpha` of the ellipse in radians.

    h : float
        The ratio :math:`h` of the inner to the outer semi-axes, in :math:`[0, 1)`.

    resolution : int
        The number of points along each axis.

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
        ``(X, Y, rho, theta, mask)``: the 2D Cartesian grids (``indexing='xy'``), the equivalent polar
        coordinates :math:`(\rho_{eq}, \theta_{eq})` of the points inside the elliptic annulus, and the 2D boolean
        mask of these points.

    """
    half_width_x = 1.05 * math.hypot(Rx * math.cos(alpha), Ry * math.sin(alpha))
    half_width_y = 1.05 * math.hypot(Rx * math.sin(alpha), Ry * math.cos(alpha))
    X, Y = numpy.meshgrid(
        numpy.linspace(x0 - half_width_x, x0 + half_width_x, resolution),
        numpy.linspace(y0 - half_width_y, y0 + half_width_y, resolution),
    )

    rho_list, theta_list = _core_cartesian_to_elliptic_annulus(
        x=X,
        y=Y,
        Rx=Rx,
        Ry=Ry,
        x0=x0,
        y0=y0,
        alpha=alpha,
        h=h,
        x_derivative=[0],
        y_derivative=[0],
    )
    rho_eq, theta_eq = rho_list[0], theta_list[0]
    mask = numpy.isfinite(rho_eq) & numpy.isfinite(theta_eq) & (rho_eq >= 0) & (rho_eq <= 1)
    return X, Y, rho_eq[mask], theta_eq[mask], mask


def _xy_display_values(
    rho: numpy.ndarray,
    theta: numpy.ndarray,
    mask: numpy.ndarray,
    n: List[int],
    m: List[int],
    x_derivative: List[int],
    y_derivative: List[int],
    Rx: float,
    Ry: float,
    alpha: float,
    h: float,
    precompute: bool,
) -> numpy.ma.MaskedArray:
    r"""
    Evaluate the polynomials on the display grid, masked outside the elliptic annulus.

    Parameters
    ----------
    rho : numpy.ndarray
        The equivalent radial coordinate :math:`\rho_{eq}` of the points inside the elliptic annulus (1D).

    theta : numpy.ndarray
        The equivalent angular coordinate :math:`\theta_{eq}` of the points inside the elliptic annulus (1D).

    mask : numpy.ndarray
        The 2D boolean mask of the points of the grid inside the elliptic annulus.

    n : List[int]
        The radial orders.

    m : List[int]
        The azimuthal frequencies. Must have the same length as ``n``.

    x_derivative : List[int]
        The orders of the derivatives with respect to :math:`x`. Must have the same length as ``n``.

    y_derivative : List[int]
        The orders of the derivatives with respect to :math:`y`. Must have the same length as ``n``.

    Rx : float
        The semi-axis :math:`R_x` of the ellipse along its x axis.

    Ry : float
        The semi-axis :math:`R_y` of the ellipse along its y axis.

    alpha : float
        The rotation angle :math:`\alpha` of the ellipse in radians.

    h : float
        The ratio :math:`h` of the inner to the outer semi-axes, in :math:`[0, 1)`.

    precompute : bool
        Passed to :func:`pyzernike.core._core_xy_polynomials`.

    Returns
    -------
    numpy.ma.MaskedArray
        An array of shape ``(Q,) + mask.shape``, masked outside the elliptic annulus.

    """
    values = numpy.full((len(n),) + mask.shape, numpy.nan)
    values[:, mask] = _core_xy_polynomials(
        rho=rho,
        theta=theta,
        n=n,
        m=m,
        x_derivative=x_derivative,
        y_derivative=y_derivative,
        Rx=Rx,
        Ry=Ry,
        alpha=alpha,
        h=h,
        precompute=precompute,
    )
    return numpy.ma.masked_invalid(values)


def _ellipse_outlines(
    Rx: float,
    Ry: float,
    x0: float,
    y0: float,
    alpha: float,
    h: float,
) -> List[Tuple[numpy.ndarray, numpy.ndarray]]:
    r"""
    Compute the outer and, if :math:`h > 0`, inner boundaries of the elliptic annulus.

    A point of the boundary of semi-axes :math:`(a, b)` in the frame of the ellipse is
    :math:`(X, Y) = (a \cos t, b \sin t)`, and in the Cartesian frame
    :math:`x = x_0 + \cos(\alpha) X - \sin(\alpha) Y` and :math:`y = y_0 + \sin(\alpha) X + \cos(\alpha) Y`.

    Parameters
    ----------
    Rx : float
        The semi-axis :math:`R_x` of the ellipse along its x axis.

    Ry : float
        The semi-axis :math:`R_y` of the ellipse along its y axis.

    x0 : float
        The x coordinate :math:`x_0` of the center of the ellipse.

    y0 : float
        The y coordinate :math:`y_0` of the center of the ellipse.

    alpha : float
        The rotation angle :math:`\alpha` of the ellipse in radians.

    h : float
        The ratio :math:`h` of the inner to the outer semi-axes, in :math:`[0, 1)`.

    Returns
    -------
    List[Tuple[numpy.ndarray, numpy.ndarray]]
        The ``(x, y)`` coordinates of each boundary.

    """
    t = numpy.linspace(0, 2 * numpy.pi, 361)
    outlines = []
    for scale in ((1.0, h) if h > 0 else (1.0,)):
        X, Y = scale * Rx * numpy.cos(t), scale * Ry * numpy.sin(t)
        outlines.append((
            x0 + math.cos(alpha) * X - math.sin(alpha) * Y,
            y0 + math.sin(alpha) * X + math.cos(alpha) * Y,
        ))
    return outlines


def _xy_title(n: int, m: int, x_derivative: int, y_derivative: int) -> str:
    r"""
    Build the LaTeX title of a Zernike polynomial plot on the elliptic annulus.

    Parameters
    ----------
    n : int
        The radial order.

    m : int
        The azimuthal frequency.

    x_derivative : int
        The order of the derivative with respect to x.

    y_derivative : int
        The order of the derivative with respect to y.

    Returns
    -------
    str
        The title :math:`Zxy_{n}^{m}, \frac{\partial^{d_x}}{\partial x^{d_x}} \frac{\partial^{d_y}}{\partial y^{d_y}}`.

    """
    return rf"$Zxy_{{{n}}}^{{{m}}},\ \frac{{\partial^{{{x_derivative}}}}}{{\partial x^{{{x_derivative}}}}}\ \frac{{\partial^{{{y_derivative}}}}}{{\partial y^{{{y_derivative}}}}}$"


def _xy_color_limits(data: numpy.ma.MaskedArray) -> Tuple[float, float]:
    r"""
    Compute the display limits of a polynomial on the elliptic annulus: :math:`[\min(\text{data}, -1), \max(\text{data}, 1)]`,
    ignoring the masked points.

    Parameters
    ----------
    data : numpy.ma.MaskedArray
        The values of the polynomial, masked outside the elliptic annulus.

    Returns
    -------
    Tuple[float, float]
        The lower and upper limits (:math:`(-1, 1)` if all the points are masked).

    """
    if data.count() == 0:
        return -1.0, 1.0
    return min(float(data.min()), -1.0), max(float(data.max()), 1.0)


def _parse_finite_float(text: str, name: str) -> float:
    r"""
    Parse a finite real number typed in a text box.

    Parameters
    ----------
    text : str
        The text of the box.

    name : str
        The name of the parameter, used in the error message.

    Returns
    -------
    float
        The parsed number.

    Raises
    ------
    ValueError
        If the text is not a finite number.

    """
    try:
        value = float(text.strip())
    except ValueError:
        raise ValueError(f"Invalid {name}: number expected.")
    if not math.isfinite(value):
        raise ValueError(f"Invalid {name}: must be finite.")
    return value


def _core_xy_display(
        n: List[int],
        m: List[int],
        x_derivative: List[int],
        y_derivative: List[int],
        Rx: float,
        Ry: float,
        x0: float,
        y0: float,
        alpha: float,
        h: float,
        precompute: bool,
        resolution: int = N_SAMPLES,
    ) -> None:
    r"""
    Display the Zernike polynomials on an elliptic annulus and their derivatives with respect to :math:`x` and
    :math:`y` in a ``matplotlib`` figure, with "Previous" and "Next" buttons to navigate through the given
    tuples ``(n, m, x_derivative, y_derivative)``.

    The polynomials are evaluated with :func:`pyzernike.core._core_xy_polynomials` in double precision on a
    regular Cartesian grid of ``resolution`` x ``resolution`` points covering the elliptic annulus (see
    :func:`_xy_display_grid`). They are drawn as colormaps, transparent outside the annulus, with the
    boundaries of the annulus. The color scale spans :math:`[\min(-1, Z), \max(1, Z)]`.

    Parameters
    ----------
    n : List[int]
        The radial orders.

    m : List[int]
        The azimuthal frequencies. Must have the same length as ``n``.

    x_derivative : List[int]
        The orders of the derivatives with respect to :math:`x`. Must have the same length as ``n``.

    y_derivative : List[int]
        The orders of the derivatives with respect to :math:`y`. Must have the same length as ``n``.

    Rx : float
        The semi-axis :math:`R_x` of the ellipse along its x axis.

    Ry : float
        The semi-axis :math:`R_y` of the ellipse along its y axis.

    x0 : float
        The x coordinate :math:`x_0` of the center of the ellipse.

    y0 : float
        The y coordinate :math:`y_0` of the center of the ellipse.

    alpha : float
        The rotation angle :math:`\alpha` of the ellipse in radians.

    h : float
        The ratio :math:`h` of the inner to the outer semi-axes, in :math:`[0, 1)`.

    precompute : bool
        Passed to :func:`pyzernike.core._core_xy_polynomials`.

    resolution : int, optional
        The number of points along each axis. Default is 200.

    Returns
    -------
    None

    """
    X, Y, rho, theta, mask = _xy_display_grid(Rx=Rx, Ry=Ry, x0=x0, y0=y0, alpha=alpha, h=h, resolution=resolution)
    values = _xy_display_values(
        rho=rho,
        theta=theta,
        mask=mask,
        n=n,
        m=m,
        x_derivative=x_derivative,
        y_derivative=y_derivative,
        Rx=Rx,
        Ry=Ry,
        alpha=alpha,
        h=h,
        precompute=precompute,
    )

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    colormap = ax.pcolormesh(X, Y, values[0], shading='auto', cmap='jet', vmin=-1, vmax=1)
    fig.colorbar(colormap, ax=ax, orientation='vertical')
    for x_outline, y_outline in _ellipse_outlines(Rx=Rx, Ry=Ry, x0=x0, y0=y0, alpha=alpha, h=h):
        ax.plot(x_outline, y_outline, color='black', linewidth=1)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    current = 0

    def _update() -> None:
        data = values[current]
        colormap.set_array(data.ravel())
        colormap.set_clim(*_xy_color_limits(data=data))
        ax.set_title(_xy_title(
            n=n[current],
            m=m[current],
            x_derivative=x_derivative[current],
            y_derivative=y_derivative[current],
        ))
        fig.canvas.draw_idle()

    def _move(step: int) -> None:
        nonlocal current
        current = (current + step) % len(n)
        _update()

    _update()

    button_previous = Button(fig.add_axes([0.3, 0.01, 0.1, 0.075]), 'Previous')
    button_next = Button(fig.add_axes([0.6, 0.01, 0.1, 0.075]), 'Next')
    button_previous.on_clicked(lambda event: _move(step=-1))
    button_next.on_clicked(lambda event: _move(step=+1))
    fig._pyzernike_buttons = (button_previous, button_next)

    plt.show()


def _core_xy_display_interactive(
        Rx: float,
        Ry: float,
        x0: float,
        y0: float,
        alpha: float,
        h: float,
        precompute: bool,
        resolution: int = N_SAMPLES,
    ) -> None:
    r"""
    Display the Zernike polynomials on an elliptic annulus and their derivatives with respect to :math:`x` and
    :math:`y` in an interactive ``matplotlib`` figure.

    The left panel of the figure contains:

    - text boxes to set :math:`n`, :math:`m` and the derivative orders with respect to :math:`x` and :math:`y`,
    - text boxes to set the parameters :math:`(R_x, R_y, x_0, y_0, \alpha, h)` of the elliptic annulus
      (:math:`\alpha` in radians),
    - a text box to set the resolution of the grid,
    - a "Generate" button,
    - a message line reporting invalid inputs.

    The plot is computed and updated only when the "Generate" button is clicked, so that the values can be
    typed without triggering a computation at each change. The polynomials are evaluated with
    :func:`pyzernike.core._core_xy_polynomials` as in :func:`_core_xy_display`: colormaps on the elliptic
    annulus, with its boundaries, and the color scale :math:`[\min(-1, Z), \max(1, Z)]`.

    Parameters
    ----------
    Rx : float
        The initial semi-axis :math:`R_x` of the ellipse along its x axis.

    Ry : float
        The initial semi-axis :math:`R_y` of the ellipse along its y axis.

    x0 : float
        The initial x coordinate :math:`x_0` of the center of the ellipse.

    y0 : float
        The initial y coordinate :math:`y_0` of the center of the ellipse.

    alpha : float
        The initial rotation angle :math:`\alpha` of the ellipse in radians.

    h : float
        The initial ratio :math:`h` of the inner to the outer semi-axes, in :math:`[0, 1)`.

    precompute : bool
        Passed to :func:`pyzernike.core._core_xy_polynomials`.

    resolution : int, optional
        The initial number of points along each axis. Default is 200.

    Returns
    -------
    None

    """
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_axes([0.40, 0.10, 0.48, 0.80])
    ax_colorbar = fig.add_axes([0.92, 0.10, 0.02, 0.80])
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    state = {"mesh": None, "colorbar": None, "outlines": []}

    fig.text(0.04, 0.94, "Zernike polynomials on an elliptic annulus", fontsize=13, fontweight='bold')
    fields = [
        ("n", 'Order n  ', '4'),
        ("m", 'Frequency m  ', '2'),
        ("x_derivative", 'x derivative  ', '0'),
        ("y_derivative", 'y derivative  ', '0'),
        ("Rx", 'Rx  ', f"{Rx:g}"),
        ("Ry", 'Ry  ', f"{Ry:g}"),
        ("x0", 'x0  ', f"{x0:g}"),
        ("y0", 'y0  ', f"{y0:g}"),
        ("alpha", 'alpha (rad)  ', f"{alpha:g}"),
        ("h", 'h  ', f"{h:g}"),
        ("resolution", 'Resolution  ', str(resolution)),
    ]
    boxes = {
        key: TextBox(fig.add_axes([0.20, 0.85 - 0.065 * index, 0.10, 0.05]), label, initial=initial)
        for index, (key, label, initial) in enumerate(fields)
    }
    button_generate = Button(fig.add_axes([0.10, 0.10, 0.20, 0.06]), 'Generate')
    message = fig.text(0.04, 0.04, "", fontsize=10, wrap=True)

    def _update(_=None) -> None:
        try:
            n = _parse_integer(text=boxes["n"].text, name="n", minimum=0)
            m = _parse_integer(text=boxes["m"].text, name="m")
            x_derivative = _parse_integer(text=boxes["x_derivative"].text, name="x derivative", minimum=0)
            y_derivative = _parse_integer(text=boxes["y_derivative"].text, name="y derivative", minimum=0)
            new_Rx = _parse_positive_float(text=boxes["Rx"].text, name="Rx")
            new_Ry = _parse_positive_float(text=boxes["Ry"].text, name="Ry")
            new_x0 = _parse_finite_float(text=boxes["x0"].text, name="x0")
            new_y0 = _parse_finite_float(text=boxes["y0"].text, name="y0")
            new_alpha = _parse_finite_float(text=boxes["alpha"].text, name="alpha")
            new_h = _parse_finite_float(text=boxes["h"].text, name="h")
            if not 0 <= new_h < 1:
                raise ValueError("Invalid h: must be in [0, 1).")
            new_resolution = _parse_integer(text=boxes["resolution"].text, name="resolution", minimum=2)
        except ValueError as error:
            message.set_text(str(error))
            message.set_color('red')
            fig.canvas.draw_idle()
            return

        if abs(m) > n or (n - m) % 2 != 0:
            message.set_text("|m| > n or n - m odd: the polynomial is zero.")
            message.set_color('darkorange')
        else:
            message.set_text("Press Generate to update the plot.")
            message.set_color('black')

        X, Y, rho, theta, mask = _xy_display_grid(
            Rx=new_Rx, Ry=new_Ry, x0=new_x0, y0=new_y0, alpha=new_alpha, h=new_h, resolution=new_resolution,
        )
        data = _xy_display_values(
            rho=rho,
            theta=theta,
            mask=mask,
            n=[n],
            m=[m],
            x_derivative=[x_derivative],
            y_derivative=[y_derivative],
            Rx=new_Rx,
            Ry=new_Ry,
            alpha=new_alpha,
            h=new_h,
            precompute=precompute,
        )[0]

        if state["mesh"] is not None:
            state["mesh"].remove()
        for line in state["outlines"]:
            line.remove()
        state["mesh"] = ax.pcolormesh(X, Y, data, shading='auto', cmap='jet')
        state["mesh"].set_clim(*_xy_color_limits(data=data))
        state["outlines"] = [
            ax.plot(x_outline, y_outline, color='black', linewidth=1)[0]
            for x_outline, y_outline in _ellipse_outlines(Rx=new_Rx, Ry=new_Ry, x0=new_x0, y0=new_y0, alpha=new_alpha, h=new_h)
        ]
        if state["colorbar"] is None:
            state["colorbar"] = fig.colorbar(state["mesh"], cax=ax_colorbar, orientation='vertical')
        else:
            state["colorbar"].update_normal(state["mesh"])

        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(Y.min(), Y.max())
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(_xy_title(n=n, m=m, x_derivative=x_derivative, y_derivative=y_derivative))
        fig.canvas.draw_idle()

    button_generate.on_clicked(_update)
    fig._pyzernike_widgets = (boxes, button_generate)

    _update()
    plt.show()