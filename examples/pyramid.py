import numpy
import numbers
from pyzernike import zernike_polynomial
import matplotlib.pyplot as plt
from matplotlib import cm

def pyramid(N: int = 5):
        """
        Plots Zernike polynomials up to a given order N in a pyramid layout.

        Parameters
        ----------
        N : int, optional
            The maximum order of Zernike polynomials to plot. The default is 5.

        Raises
        ------
        ValueError
            If N is not a non-negative integer.
        """
        if not isinstance(N, numbers.Integral) or N < 0:
            raise ValueError("N must be a non-negative integer.")
        
        fs = 18
        fs_sub = 8
        ax_spacing = 0.1
        title_spacing = 0.4
        sub_spacing = 0.10
        
        # Create a new figure
        Nb_rows = N + 1 + title_spacing + sub_spacing
        Nb_cols = N + 1

        fig = plt.figure(figsize=(Nb_cols, Nb_rows))

        rho = numpy.linspace(0, 1, 400)
        theta = numpy.linspace(0, 2 * numpy.pi, 400)
        rho, theta = numpy.meshgrid(rho, theta)
        X, Y = rho * numpy.cos(theta), rho * numpy.sin(theta)

        # Configure the colormap normalization for consistent color scaling
        norm = cm.colors.Normalize(vmin=-1, vmax=1)
        mappable = cm.ScalarMappable(norm=norm, cmap=cm.seismic)

        # Iterate through the orders of Zernike polynomials
        for n in range(N + 1):
            for m in range(-n, n + 1, 2):
                
                # Coordinates for the lower left corner of the subplot
                lower_bottom_x = ax_spacing + (N + m) / 2
                lower_bottom_y = sub_spacing + ax_spacing + N - n
                size = 1 - 2 * ax_spacing

                # Scale the coordinates to fit the subplot grid
                scale_lower_bottom_x = lower_bottom_x / Nb_cols
                scale_lower_bottom_y = lower_bottom_y / Nb_rows
                scale_size_x = size / Nb_cols
                scale_size_y = size / Nb_rows

                # Create the subplot axes
                ax = fig.add_axes([scale_lower_bottom_x, scale_lower_bottom_y, scale_size_x, scale_size_y])

                # Calculate the Zernike polynomial
                Z = zernike_polynomial(rho, theta, n=[n], m=[m], rho_derivative=[0], theta_derivative=[0])[0]

                # Plot the Zernike polynomial
                im = ax.pcolormesh(X, Y, Z, cmap=cm.seismic, norm=norm)
                ax.text(0.5, -0.1, f"Z({n},{m})", fontsize=fs_sub, ha='center', va='top', transform=ax.transAxes)
                ax.set_aspect('equal')
                ax.axis('off')

        fig.suptitle(f"Zernike Polynomials", fontsize=fs)
        plt.show()

if __name__ == "__main__":
    pyramid(N=5)