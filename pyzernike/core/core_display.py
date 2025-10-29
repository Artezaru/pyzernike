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
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from .core_polynomial import core_polynomial
from .core_corresponding_signed_integer_type import core_corresponding_signed_integer_type

def core_display(
        n: numpy.array,
        m: numpy.array,
        rho_derivative: numpy.array,
        theta_derivative: Optional[numpy.array],
        flag_radial: bool,
        precompute: bool,
        float_type: type[numpy.floating]
    ) -> None:
    r"""
    Display Zernike polynomials for given `n`, `m`, `rho_derivative`, and `theta_derivative` values.

    .. warning::

        This method is a core function of ``pyzernike`` that is not designed to be use by the users directly.
        Please use the high level functions.

    .. seealso::

        - :func:`pyzernike.radial_display` for the radial Zernike polynomial display.
        - :func:`pyzernike.zernike_display` for the full Zernike polynomial display.
        - The page :doc:`../../mathematical_description` in the documentation for the mathematical description of the Zernike polynomials.

    - ``n``, ``m``, ``rho_derivative`` and ``theta_derivative`` are expected to be sequences of integers of the same length and valid values.

    The displays are including in a interactive matplotlib figure with buttons to navigate through the different Zernike polynomials.

    Parameters
    ----------    
    n : numpy.array[numpy.integer]
        The orders of the Zernike polynomials to compute. Must be a 1D array of integers of type compatible with ``float_type``.

    m : numpy.array[numpy.integer]
        The degrees of the Zernike polynomials. Must be a 1D array of integers of type compatible with ``float_type``.

    numpy.array[numpy.integer]
        The orders of the derivatives with respect to rho. Must be a 1D array of integers of type compatible with ``float_type``.

    theta_derivative : Optional[numpy.array[numpy.integer]]
        The orders of the derivatives with respect to theta. Must be None if ``flag_radial`` is True. Otherwise, must be a 1D array of integers of type compatible with ``float_type``.

    flag_radial : bool
        If True, computes the sets for radial polynomials only (no angular part). The output sine and cosine frequency sets will be empty.
        If False, computes the sets for full Zernike polynomials (including angular part).

    precompute : bool
        If True, the useful terms for the Zernike polynomials are precomputed to optimize the computation.
        This is useful when computing multiple Zernike polynomials with the same `rho` and `theta` values.
        If False, the useful terms are computed on-the-fly for each polynomial, which may be slower but avoid memory overhead.

    float_type : type[numpy.floating]
        The floating point type used for the computations (e.g., numpy.float32, numpy.float64).

    Returns
    -------
    None
    """
    # Get the corresponding integer types
    int_type = core_corresponding_signed_integer_type(float_type)

    # Fast assertions on the inputs
    assert issubclass(float_type, numpy.floating), "[pyzernike-core] float_type must be a numpy floating point type."
    assert isinstance(n, numpy.ndarray) and n.ndim == 1 and numpy.issubdtype(n.dtype, int_type), "[pyzernike-core] n must be a 1D numpy array of integers of type compatible with float_type."
    assert isinstance(m, numpy.ndarray) and m.ndim == 1 and numpy.issubdtype(m.dtype, int_type), "[pyzernike-core] m must be a 1D numpy array of integers of type compatible with float_type."
    assert isinstance(rho_derivative, numpy.ndarray) and rho_derivative.ndim == 1 and numpy.issubdtype(rho_derivative.dtype, int_type), "[pyzernike-core] rho_derivative must be a 1D numpy array of integers of type compatible with float_type."
    assert isinstance(flag_radial, bool), "[pyzernike-core] flag_radial must be a boolean."
    assert flag_radial or (isinstance(theta_derivative, numpy.ndarray) and theta_derivative.ndim == 1 and numpy.issubdtype(theta_derivative.dtype, int_type)), "[pyzernike-core] theta_derivative must be a 1D numpy array of integers of type compatible with float_type when flag_radial is False."
    assert not flag_radial or theta_derivative is None, "[pyzernike-core] theta_derivative must be None when flag_radial is True."
    assert n.size == m.size == rho_derivative.size and (flag_radial or n.size == theta_derivative.size), "[pyzernike-core] n, m, rho_derivative and theta_derivative (if flag_radial is False) must have the same size."

    # Compute the Zernike polynomial values
    rho = numpy.linspace(0, 1.0, 200, dtype=float_type)
    theta = numpy.linspace(0, 2 * numpy.pi, 200, dtype=float_type)
    Rho, Theta = numpy.meshgrid(rho, theta, indexing='ij')
    
    # Compute the values of the Zernike polynomials
    if flag_radial:
        # Radial Zernike polynomial
        Values = core_polynomial(
            rho=rho,
            theta=None,
            n=n,
            m=m,
            rho_derivative=rho_derivative,
            theta_derivative=None,
            flag_radial=True,
            precompute=precompute,
            float_type=float_type,
        )
    else:
        # Full Zernike polynomial
        Values = core_polynomial(
            rho=Rho,
            theta=Theta,
            n=n,
            m=m,
            rho_derivative=rho_derivative,
            theta_derivative=theta_derivative,
            flag_radial=False,
            precompute=precompute,
            float_type=float_type,
        )

    # Indexes for the current plot
    current_plot: list[int] = [0]  # List to hold the current index for mutability

    # Prepare the radial coordinates for plotting
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'} if not flag_radial else {})
    plt.subplots_adjust(bottom=0.2)

    # Plotting the data
    if flag_radial:
        plot_radial, = ax.plot(rho, Values[0])
        title = rf"$\mathrm{{Radial\ Zernike}}\ R_{{{n[0]}}}^{{{m[0]}}}(\rho),\ \frac{{d^{{{rho_derivative[0]}}}}}{{d\rho^{{{rho_derivative[0]}}}}}$"
        ax.set_ylabel('Amplitude')
        ax.set_xlabel('ρ (radial coordinate)')
        ax.set_xlim(0, 1)
        ax.set_ylim(-1, 1)
        ax.grid(True)
        pcm_zernike = None # For radial plots, we don't use pcolormesh
    else:
        pcm_zernike = ax.pcolormesh(Theta, Rho, Values[0], shading='auto', cmap='jet', vmin=-1, vmax=1)
        colorbar = fig.colorbar(pcm_zernike, ax=ax, orientation='vertical')
        title = rf"$Z_{{{n[0]}}}^{{{m[0]}}},\ \frac{{\partial^{rho_derivative[0]}}}{{\partial \rho^{rho_derivative[0]}}}\ \frac{{\partial^{theta_derivative[0]}}}{{\partial \theta^{theta_derivative[0]}}}$"
        plot_radial = None  # For full Zernike plots, we don't use a radial plot

    ax.set_title(title)

    # Function to update the plot with new data
    def update_plot_data(index, fig, ax, n, m, rho_derivative, theta_derivative, Values, flag_radial, plot_radial, pcm_zernike):
        # Extract the data for the current index
        data = Values[index]

        if flag_radial:
            # Plot radial Zernike polynomial
            plot_radial.set_ydata(data)
            title = rf"$\mathrm{{Radial\ Zernike}}\ R_{{{n[index]}}}^{{{m[index]}}}(\rho),\ \frac{{d^{{{rho_derivative[index]}}}}}{{d\rho^{{{rho_derivative[index]}}}}}$"
            ax.set_ylim(min(numpy.min(data), -1), max(numpy.max(data), 1))
        else:
            # Plot full Zernike polynomial
            pcm_zernike.set_array(data.flatten())
            title = rf"$Z_{{{n[index]}}}^{{{m[index]}}},\ \frac{{\partial^{rho_derivative[index]}}}{{\partial \rho^{rho_derivative[index]}}}\ \frac{{\partial^{theta_derivative[index]}}}{{\partial \theta^{theta_derivative[index]}}}$"
            pcm_zernike.set_clim(min(numpy.min(data), -1), max(numpy.max(data), 1))

        ax.set_title(title)
        fig.canvas.draw_idle()

    update_plot_data(current_plot[0], fig, ax, n, m, rho_derivative, theta_derivative, Values, flag_radial, plot_radial, pcm_zernike)

    # Create buttons for navigation
    axprev = plt.axes([0.3, 0.01, 0.1, 0.075])
    axnext = plt.axes([0.6, 0.01, 0.1, 0.075])
    bprev = Button(axprev, 'Previous')
    bnext = Button(axnext, 'Next')

    def next(event, fig, ax, n, m, rho_derivative, theta_derivative, Values, flag_radial, plot_radial, pcm_zernike):
        current_plot[0] = (current_plot[0] + 1) % len(n)
        update_plot_data(current_plot[0], fig, ax, n, m, rho_derivative, theta_derivative, Values, flag_radial, plot_radial, pcm_zernike)

    next_event = lambda event: next(event, fig, ax, n, m, rho_derivative, theta_derivative, Values, flag_radial, plot_radial, pcm_zernike)

    def prev(event, fig, ax, n, m, rho_derivative, theta_derivative, Values, flag_radial, plot_radial, pcm_zernike):
        current_plot[0] = (current_plot[0] - 1) % len(n)
        update_plot_data(current_plot[0], fig, ax, n, m, rho_derivative, theta_derivative, Values, flag_radial, plot_radial, pcm_zernike)

    prev_event = lambda event: prev(event, fig, ax, n, m, rho_derivative, theta_derivative, Values, flag_radial, plot_radial, pcm_zernike)

    bnext.on_clicked(next_event)
    bprev.on_clicked(prev_event)

    plt.show()