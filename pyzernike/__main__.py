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

from pyzernike import zernike_display, radial_display
from pyzernike.core import core_display_interactive
import argparse
import numpy

def __main__() -> None:
    r"""
    Main entry point of the package.

    This method contains the script to run if the user enter the name of the package on the command line. 

    .. code-block:: console
    
        pyzernike

    This will display a set of Zernike polynomials in an interactive matplotlib figure.

    .. code-block:: console

        pyzernike -r -n 3
    
    - flag `-r` or `--radial` will display the radial Zernike polynomials instead of the full Zernike polynomials.
    - flag `-n` or `--n` will specify the maximum order of the Zernike polynomials to display. If not specified, the default value is 5
    - flag `-dr` or `--rho_derivative` can be used to specify the radial derivative of the Zernike polynomials. If not specified, the default value is 0 for all polynomials.
    - flag `-dt` or `--theta_derivative` can be used to specify the angular derivative of the Zernike polynomials. If not specified, the default value is 0 for all polynomials.
    - flag `-i` or `--interactive` can be used to launch the interactive display window. If not specified, the default behavior is to launch the interactive display.
    - flag `-h` or `--help` can be used to display the help message.
        
    """
    # Argument parser
    parser = argparse.ArgumentParser(description="Display Zernike polynomials.")
    parser.add_argument(
        '-r', '--radial', action='store_true',
        help="Display radial Zernike polynomials instead of full Zernike polynomials."
    )
    parser.add_argument(
        '-n', '--n', type=int, default=5,
        help="Maximum order of the Zernike polynomials to display (default: 5)."
    )
    parser.add_argument(
        '-dr', '--rho_derivative', type=int, default=0,
        help="Radial derivative of the Zernike polynomials to display (default: 0 for all polynomials)."
    )
    parser.add_argument(
        '-dt', '--theta_derivative', type=int, default=0,
        help="Angular derivative of the Zernike polynomials to display (default: 0 for all polynomials)."
    )
    parser.add_argument(
        '-i', '--interactive', action='store_true',
        help="Launch the interactive display window."
    )

    args = parser.parse_args()

    Nzer = args.n
    dr = args.rho_derivative
    dt = args.theta_derivative
    interactive = args.interactive
    
    if interactive:
        core_display_interactive(numpy.float64)

    if not isinstance(Nzer, int) or Nzer <= 0:
        raise ValueError("The maximum order of the Zernike polynomials must be a non-negative integer.")
    if not isinstance(dr, int) or dr < 0:
        raise ValueError("The radial derivative must be a positive integer.")
    if not isinstance(dt, int) or dt < 0:
        raise ValueError("The angular derivative must be a non-negative integer.")

    list_n = []
    list_m = []
    if args.radial:
        # For radial Zernike polynomials, m is always even
        for n in range(0, Nzer + 1):
            for m in range(0 if n%2 == 0 else 1, n + 1, 2):
                list_n.append(n)
                list_m.append(m)
    
    else:
        # For full Zernike polynomials, m can be both even and odd
        for n in range(0, Nzer + 1):
            for m in range(-n, n + 1, 2):
                list_n.append(n)
                list_m.append(m)
    
    # Display the Zernike polynomials
    if args.radial:
        radial_display(n=list_n, m=list_m, rho_derivative=[args.rho_derivative for _ in range(len(list_n))])
    else:
        zernike_display(n=list_n, m=list_m, rho_derivative=[args.rho_derivative for _ in range(len(list_n))], theta_derivative=[args.theta_derivative for _ in range(len(list_n))])


def __main_gui__() -> None:
    r"""
    Graphical user interface entry point of the package.

    This method contains the script to run if the user enter the name of the package on the command line with the ``gui`` extension.

    .. code-block:: console

        pyzernike-gui
        
    """
    raise NotImplementedError("The graphical user interface entry point is not implemented yet.")

