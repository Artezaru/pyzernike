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

from typing import List, Optional, Tuple

import numpy
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, TextBox

from ..core.core_polynomial import _core_polynomials


N_SAMPLES = 200


def _mapping_label(rho_power: float) -> str:
    r"""
    Build the LaTeX label of the display mapping :math:`\rho = s^{p}`, or an empty string if :math:`p = 1`.

    Parameters
    ----------
    rho_power : float
        The strictly positive exponent :math:`p`.

    Returns
    -------
    str
        The label.

    """
    if rho_power == 1:
        return ""
    return rf"$\rho = s^{{{rho_power:g}}}$"


def _display_grid(resolution: int, rho_power: float) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    r"""
    Build the display grid and the corresponding evaluation points.

    The display coordinate :math:`s` spans :math:`[0, 1]` with ``resolution`` points (and :math:`\theta`
    spans :math:`[0, 2\pi]` with ``resolution`` points). The polynomials are evaluated at :math:`\rho = s^{p}`
    with :math:`p` = ``rho_power`` > 0, which also spans :math:`[0, 1]`.

    Parameters
    ----------
    resolution : int
        The number of points along each axis.

    rho_power : float
        The strictly positive exponent :math:`p`.

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
        ``(s, rho, S, Rho, Theta)``: the 1D display coordinate, the 1D evaluation points, and the 2D grids
        of the display coordinate, of the evaluation points and of the angles (``indexing='ij'``).

    """
    s = numpy.linspace(0, 1.0, resolution)
    theta = numpy.linspace(0, 2 * numpy.pi, resolution)
    S, Theta = numpy.meshgrid(s, theta, indexing='ij')
    return s, s ** rho_power, S, S ** rho_power, Theta


def _radial_title(n: int, m: int, rho_derivative: int) -> str:
    r"""
    Build the LaTeX title of a radial polynomial plot.

    Parameters
    ----------
    n : int
        The radial order.

    m : int
        The azimuthal frequency.

    rho_derivative : int
        The order of the derivative with respect to rho.

    Returns
    -------
    str
        The title :math:`R_{n}^{m}(\rho), \frac{d^{a}}{d\rho^{a}}`.

    """
    return rf"$\mathrm{{Radial\ Zernike}}\ R_{{{n}}}^{{{m}}}(\rho),\ \frac{{d^{{{rho_derivative}}}}}{{d\rho^{{{rho_derivative}}}}}$"


def _zernike_title(n: int, m: int, rho_derivative: int, theta_derivative: int) -> str:
    r"""
    Build the LaTeX title of a Zernike polynomial plot.

    Parameters
    ----------
    n : int
        The radial order.

    m : int
        The azimuthal frequency.

    rho_derivative : int
        The order of the derivative with respect to rho.

    theta_derivative : int
        The order of the derivative with respect to theta.

    Returns
    -------
    str
        The title :math:`Z_{n}^{m}, \frac{\partial^{a}}{\partial \rho^{a}} \frac{\partial^{b}}{\partial \theta^{b}}`.

    """
    return rf"$Z_{{{n}}}^{{{m}}},\ \frac{{\partial^{{{rho_derivative}}}}}{{\partial \rho^{{{rho_derivative}}}}}\ \frac{{\partial^{{{theta_derivative}}}}}{{\partial \theta^{{{theta_derivative}}}}}$"


def _color_limits(data: numpy.ndarray) -> tuple:
    r"""
    Compute the display limits of a polynomial: :math:`[\min(\text{data}, -1), \max(\text{data}, 1)]`.

    Parameters
    ----------
    data : numpy.ndarray
        The values of the polynomial.

    Returns
    -------
    tuple
        The lower and upper limits.

    """
    return min(numpy.min(data), -1), max(numpy.max(data), 1)


def _core_display(
        n: List[int],
        m: List[int],
        rho_derivative: List[int],
        theta_derivative: Optional[List[int]],
        flag_radial: bool,
        precompute: bool,
        resolution: int = N_SAMPLES,
        rho_power: float = 1.0,
    ) -> None:
    r"""
    Display the Zernike polynomials :math:`Z_{n}^{m}(\rho, \theta)` (or the radial polynomials
    :math:`R_{n}^{m}(\rho)` if ``flag_radial`` is True) and their derivatives in a ``matplotlib``
    figure, with "Previous" and "Next" buttons to navigate through the given tuples
    ``(n, m, rho_derivative, theta_derivative)``.

    The polynomials are evaluated with :func:`pyzernike.core._core_polynomials` in double precision on
    a regular grid of ``resolution`` points for the display coordinate :math:`s \in [0, 1]` (and
    ``resolution`` points for :math:`\theta \in [0, 2\pi]`), at the radial coordinate :math:`\rho = s^{p}`
    with :math:`p` = ``rho_power`` > 0. With :math:`p < 1` (e.g. :math:`p = 1/2`), the region near the edge
    of the disk, where the high order polynomials oscillate the most, is enlarged; with :math:`p > 1`, the
    center of the disk is enlarged. The derivatives are the
    derivatives with respect to :math:`\rho` (not :math:`s`), evaluated at :math:`\rho = s^{p}`.

    - Radial polynomials are drawn as curves :math:`\rho \mapsto R_{n}^{m}(\rho)`.
    - Zernike polynomials are drawn as colormaps on the unit disk (polar axes).

    The vertical axis (radial) or the color scale (Zernike) spans :math:`[\min(-1, Z), \max(1, Z)]`,
    so that polynomials of different orders are displayed on comparable scales.

    Parameters
    ----------
    n : List[int]
        The radial orders.

    m : List[int]
        The azimuthal frequencies. Must have the same length as ``n``.

    rho_derivative : List[int]
        The orders of the derivatives with respect to rho. Must have the same length as ``n``.

    theta_derivative : Optional[List[int]]
        The orders of the derivatives with respect to theta. Must have the same length as ``n``.
        Must be None if ``flag_radial`` is True.

    flag_radial : bool
        If True, displays the radial polynomials only. If False, displays the full Zernike polynomials.

    precompute : bool
        Passed to :func:`pyzernike.core._core_polynomials`.

    resolution : int, optional
        The number of points along each axis. Default is 200.

    rho_power : float, optional
        The strictly positive exponent :math:`p` of the display mapping :math:`\rho = s^{p}`. Default is 1 (no mapping).

    Returns
    -------
    None

    """
    s, rho, S, Rho, Theta = _display_grid(resolution=resolution, rho_power=rho_power)
    mapping = _mapping_label(rho_power=rho_power)

    if flag_radial:
        values = _core_polynomials(
            rho=rho,
            theta=None,
            n=n,
            m=m,
            rho_derivative=rho_derivative,
            theta_derivative=None,
            flag_radial=True,
            precompute=precompute,
        )
    else:
        values = _core_polynomials(
            rho=Rho,
            theta=Theta,
            n=n,
            m=m,
            rho_derivative=rho_derivative,
            theta_derivative=theta_derivative,
            flag_radial=False,
            precompute=precompute,
        )

    fig, ax = plt.subplots(subplot_kw={} if flag_radial else {'projection': 'polar'})
    plt.subplots_adjust(bottom=0.2)

    if flag_radial:
        curve, = ax.plot(s, values[0])
        ax.set_xlabel(f"s  ({mapping})" if mapping else 'ρ (radial coordinate)')
        ax.set_ylabel('Amplitude')
        ax.set_xlim(0, 1)
        ax.grid(True)
    else:
        colormap = ax.pcolormesh(Theta, S, values[0], shading='auto', cmap='jet', vmin=-1, vmax=1)
        ax.set_ylim(0, 1)  # radial limits: the cells centered on s = 0 would extend them below 0 (white disk at the center)
        fig.colorbar(colormap, ax=ax, orientation='vertical')
        if mapping:
            ax.set_xlabel(f"radial axis: s  ({mapping})")

    current = 0

    def _update() -> None:
        data = values[current]
        if flag_radial:
            curve.set_ydata(data)
            ax.set_ylim(*_color_limits(data=data))
            ax.set_title(_radial_title(n=n[current], m=m[current], rho_derivative=rho_derivative[current]))
        else:
            colormap.set_array(data.flatten())
            colormap.set_clim(*_color_limits(data=data))
            ax.set_title(_zernike_title(
                n=n[current],
                m=m[current],
                rho_derivative=rho_derivative[current],
                theta_derivative=theta_derivative[current],
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



def _parse_integer(text: str, name: str, minimum: Optional[int] = None) -> int:
    r"""
    Parse an integer typed in a text box.

    Parameters
    ----------
    text : str
        The text of the box. Integers written as floats (e.g. ``"3.0"``) are accepted.

    name : str
        The name of the parameter, used in the error message.

    minimum : Optional[int], optional
        The minimum accepted value. Default is None (no minimum).

    Returns
    -------
    int
        The parsed integer.

    Raises
    ------
    ValueError
        If the text is not an integer or is lower than ``minimum``.

    """
    try:
        value = float(text.strip())
    except ValueError:
        raise ValueError(f"Invalid {name}: integer expected.")
    if not value.is_integer():
        raise ValueError(f"Invalid {name}: integer expected.")
    if minimum is not None and value < minimum:
        raise ValueError(f"Invalid {name}: must be >= {minimum}.")
    return int(value)


def _parse_positive_float(text: str, name: str) -> float:
    r"""
    Parse a strictly positive real number typed in a text box.

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
        If the text is not a finite number strictly greater than 0.

    """
    try:
        value = float(text.strip())
    except ValueError:
        raise ValueError(f"Invalid {name}: number expected.")
    if not numpy.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid {name}: must be > 0.")
    return value


def _core_display_interactive(precompute: bool, resolution: int = N_SAMPLES, rho_power: float = 1.0) -> None:
    r"""
    Display the Zernike polynomials :math:`Z_{n}^{m}(\rho, \theta)` or the radial polynomials
    :math:`R_{n}^{m}(\rho)` and their derivatives in an interactive ``matplotlib`` figure.

    The left panel of the figure contains:

    - a selector to switch between the full Zernike polynomials and the radial polynomials,
    - text boxes to set :math:`n`, :math:`m`, the derivative order with respect to :math:`\rho` and the
      derivative order with respect to :math:`\theta` (ignored for the radial polynomials),
    - text boxes to set the resolution of the grid and the exponent ``rho_power`` > 0 of the display
      mapping :math:`\rho = s^{p}` (see :func:`_core_display`),
    - a "Generate" button,
    - a message line reporting invalid inputs.

    The plot is computed and updated only when the "Generate" button is clicked, so that the values can be
    typed without triggering a computation at each change. The selector only shows or hides the text box of
    the derivative with respect to :math:`\theta`. The polynomials are evaluated with :func:`pyzernike.core._core_polynomials` as in :func:`_core_display`:
    curves for the radial polynomials, colormaps on the unit disk for the Zernike polynomials, with the
    scale :math:`[\min(-1, Z), \max(1, Z)]`.

    Parameters
    ----------
    precompute : bool
        Passed to :func:`pyzernike.core._core_polynomials`.

    resolution : int, optional
        The initial number of points along each axis. Default is 200.

    rho_power : float, optional
        The initial strictly positive exponent of the display mapping :math:`\rho = s^{p}`. Default is 1.

    Returns
    -------
    None

    """
    fig = plt.figure(figsize=(11, 6.5))
    ax_radial = fig.add_axes([0.40, 0.12, 0.48, 0.76])
    ax_zernike = fig.add_axes([0.40, 0.12, 0.48, 0.76], projection='polar')
    ax_colorbar = fig.add_axes([0.92, 0.12, 0.02, 0.76])

    curve, = ax_radial.plot([0, 1], [0, 0])
    ax_radial.set_ylabel('Amplitude')
    ax_radial.set_xlim(0, 1)
    ax_radial.grid(True)

    state = {"grid": None, "colormap": None, "colorbar": None}

    fig.text(0.05, 0.93, "Zernike polynomials", fontsize=13, fontweight='bold')
    selector = RadioButtons(fig.add_axes([0.05, 0.77, 0.22, 0.13]), ['Zernike Z(ρ, θ)', 'Radial R(ρ)'])
    boxes = {
        "n": TextBox(fig.add_axes([0.20, 0.66, 0.08, 0.055]), 'Order n  ', initial='4'),
        "m": TextBox(fig.add_axes([0.20, 0.58, 0.08, 0.055]), 'Frequency m  ', initial='2'),
        "rho_derivative": TextBox(fig.add_axes([0.20, 0.50, 0.08, 0.055]), 'ρ derivative  ', initial='0'),
        "theta_derivative": TextBox(fig.add_axes([0.20, 0.42, 0.08, 0.055]), 'θ derivative  ', initial='0'),
        "resolution": TextBox(fig.add_axes([0.20, 0.30, 0.08, 0.055]), 'Resolution  ', initial=str(resolution)),
        "rho_power": TextBox(fig.add_axes([0.20, 0.22, 0.08, 0.055]), 'ρ power  ', initial=f"{rho_power:g}"),
    }
    button_generate = Button(fig.add_axes([0.10, 0.12, 0.18, 0.06]), 'Generate')
    message = fig.text(0.05, 0.05, "", fontsize=10, wrap=True)

    def _set_grid(new_resolution: int, new_rho_power: float) -> None:
        if state["grid"] is not None and state["grid"][:2] == (new_resolution, new_rho_power):
            return
        s, rho, S, Rho, Theta = _display_grid(resolution=new_resolution, rho_power=new_rho_power)
        state["grid"] = (new_resolution, new_rho_power, s, rho, S, Rho, Theta)

        if state["colormap"] is not None:
            state["colormap"].remove()
        state["colormap"] = ax_zernike.pcolormesh(Theta, S, numpy.zeros_like(S), shading='auto', cmap='jet', vmin=-1, vmax=1)
        ax_zernike.set_ylim(0, 1)  # radial limits: the successive meshes would extend them below 0 (white disk at the center)
        if state["colorbar"] is None:
            state["colorbar"] = fig.colorbar(state["colormap"], cax=ax_colorbar, orientation='vertical')
        else:
            state["colorbar"].update_normal(state["colormap"])

        mapping = _mapping_label(rho_power=new_rho_power)
        ax_radial.set_xlabel(f"s  ({mapping})" if mapping else 'ρ (radial coordinate)')
        ax_zernike.set_xlabel(f"radial axis: s  ({mapping})" if mapping else '')

    def _select(_=None) -> None:
        flag_radial = selector.value_selected.startswith('Radial')
        boxes["theta_derivative"].ax.set_visible(not flag_radial)
        message.set_text("Press Generate to update the plot.")
        message.set_color('black')
        fig.canvas.draw_idle()

    def _update(_=None) -> None:
        flag_radial = selector.value_selected.startswith('Radial')
        ax_radial.set_visible(flag_radial)
        ax_zernike.set_visible(not flag_radial)
        ax_colorbar.set_visible(not flag_radial)
        boxes["theta_derivative"].ax.set_visible(not flag_radial)

        try:
            n = _parse_integer(text=boxes["n"].text, name="n", minimum=0)
            m = _parse_integer(text=boxes["m"].text, name="m")
            rho_derivative = _parse_integer(text=boxes["rho_derivative"].text, name="ρ derivative", minimum=0)
            theta_derivative = 0 if flag_radial else _parse_integer(text=boxes["theta_derivative"].text, name="θ derivative", minimum=0)
            new_resolution = _parse_integer(text=boxes["resolution"].text, name="resolution", minimum=2)
            new_rho_power = _parse_positive_float(text=boxes["rho_power"].text, name="ρ power")
            if flag_radial and m < 0:
                raise ValueError("Invalid m: must be >= 0 for the radial polynomials.")
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

        _set_grid(new_resolution=new_resolution, new_rho_power=new_rho_power)
        _, _, s, rho, S, Rho, Theta = state["grid"]

        if flag_radial:
            data = _core_polynomials(
                rho=rho,
                theta=None,
                n=[n],
                m=[m],
                rho_derivative=[rho_derivative],
                theta_derivative=None,
                flag_radial=True,
                precompute=precompute,
            )[0]
            curve.set_data(s, data)
            ax_radial.set_ylim(*_color_limits(data=data))
            ax_radial.set_title(_radial_title(n=n, m=m, rho_derivative=rho_derivative))
        else:
            data = _core_polynomials(
                rho=Rho,
                theta=Theta,
                n=[n],
                m=[m],
                rho_derivative=[rho_derivative],
                theta_derivative=[theta_derivative],
                flag_radial=False,
                precompute=precompute,
            )[0]
            state["colormap"].set_array(data.flatten())
            state["colormap"].set_clim(*_color_limits(data=data))
            ax_zernike.set_title(_zernike_title(n=n, m=m, rho_derivative=rho_derivative, theta_derivative=theta_derivative))

        fig.canvas.draw_idle()

    selector.on_clicked(_select)
    button_generate.on_clicked(_update)
    fig._pyzernike_widgets = (selector, boxes, button_generate)

    _update()
    plt.show()