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
from typing import Sequence, List
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from .core_polynomial import core_polynomial

def core_display(
        n: Sequence[int], 
        m: Sequence[int], 
        rho_derivative: Sequence[int], 
        theta_derivative: Sequence[int],
        flag_radial: bool,
    ) -> None:
    r"""

    .. warning::

        This method is a core function of ``pyzernike`` that displays Zernike polynomials for given `n`, `m`, `rho_derivative`, and `theta_derivative` values.
        It is not a method designed to be use by the users directly, but rather a helper function for the optimized display.

        ``n``, ``m``, ``rho_derivative`` and ``theta_derivative`` are expected to be sequences of integers with the same length and valid values.

    .. seealso::

        - :func:`pyzernike.radial_display` for the radial Zernike polynomial display.
        - :func:`pyzernike.zernike_display` for the full Zernike polynomial display.

    The displays are including in a interactive matplotlib figure with buttons to navigate through the different Zernike polynomials.

    Parameters
    ----------    
    n : Sequence[int]
        A list of integers containing the order `n` of each radial Zernike polynomial to display.

    m : Sequence[int]
        A list of integers containing the degree `m` of each radial Zernike polynomial to display.

    rho_derivative : Sequence[int]
        A list of integers containing the order of the radial derivative to display for each radial Zernike polynomial.
        If `rho_derivative` is None, no radial derivative is displayed.

    theta_derivative : Sequence[int]
        A list of integers containing the order of the angular derivative to display for each Zernike polynomial.
        If `theta_derivative` is None, no angular derivative is displayed. ONLY USED IF `flag_radial` IS False.

    flag_radial : bool
        If True, the radial Zernike polynomial :math:`R_{n}^{m}(\rho)` is displayed instead of the full Zernike polynomial :math:`Z_{n}^{m}(\rho, \theta)`.
        If False, the full Zernike polynomial :math:`Z_{n}^{m}(\rho, \theta)` is displayed, which includes the angular part with the cosine and sine terms.

    Returns
    -------
    None
    """
    # Compute the Zernike polynomial values
    rho = numpy.linspace(0, 1.0, 200)
    theta = numpy.linspace(0, 2 * numpy.pi, 200)
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
        )

    # Indexes for the current plot
    current_plot = [0]  # List to hold the current index for mutability

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
        else:
            # Plot full Zernike polynomial
            pcm_zernike.set_array(data.flatten())
            title = rf"$Z_{{{n[index]}}}^{{{m[index]}}},\ \frac{{\partial^{rho_derivative[index]}}}{{\partial \rho^{rho_derivative[index]}}}\ \frac{{\partial^{theta_derivative[index]}}}{{\partial \theta^{theta_derivative[index]}}}$"
        
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