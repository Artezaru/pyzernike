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

import argparse

from pyzernike import display_interactive, radial_display, zernike_display, zernike_index_to_order, zernike_polynomials_count


def _positive_float(text: str) -> float:
    value = float(text)
    if not value > 0:
        raise argparse.ArgumentTypeError("must be a strictly positive number.")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pyzernike", description="Display Zernike polynomials.")
    parser.add_argument(
        '-r', '--radial', action='store_true',
        help="Display the radial Zernike polynomials instead of the full Zernike polynomials."
    )
    parser.add_argument(
        '-n', '--n', type=int, default=5,
        help="Maximum order of the Zernike polynomials to display (default: 5)."
    )
    parser.add_argument(
        '-dr', '--rho_derivative', type=int, default=0,
        help="Order of the derivative with respect to rho (default: 0)."
    )
    parser.add_argument(
        '-dt', '--theta_derivative', type=int, default=0,
        help="Order of the derivative with respect to theta, ignored with --radial (default: 0)."
    )
    parser.add_argument(
        '--resolution', type=int, default=200,
        help="Number of points along each axis of the display grid (default: 200)."
    )
    parser.add_argument(
        '--rho_power', type=_positive_float, default=1.0,
        help="Display mapping rho = s**rho_power, e.g. 0.5 to enlarge the edge of the disk (default: 1)."
    )
    parser.add_argument(
        '-i', '--interactive', action='store_true',
        help="Launch the interactive display window (only --resolution and --rho_power are used)."
    )
    return parser


def __main__() -> None:
    r"""
    Main entry point of the package.

    This method contains the script to run if the user enters the name of the package on the command line.
    It displays all the Zernike polynomials (or the radial polynomials) up to a given order, with the
    "Previous" and "Next" buttons of the figure to navigate through them.

    .. code-block:: console

        pyzernike -r -n 3

    - flag ``-r`` or ``--radial`` displays the radial Zernike polynomials instead of the full Zernike polynomials.
    - flag ``-n {N}`` or ``--n {N}`` sets the maximum order of the polynomials to display (default: 5).
    - flag ``-dr {D}`` or ``--rho_derivative {D}`` sets the order of the derivative with respect to :math:`\rho` (default: 0).
    - flag ``-dt {D}`` or ``--theta_derivative {D}`` sets the order of the derivative with respect to :math:`\theta` (default: 0, ignored with ``-r``).
    - flag ``--resolution {R}`` sets the number of points along each axis of the display grid (default: 200).
    - flag ``--rho_power {P}`` sets the display mapping :math:`\rho = s^{P}` (default: 1, see :func:`pyzernike.zernike_display`).
    - flag ``-i`` or ``--interactive`` launches the interactive display window (see :func:`pyzernike.display_interactive`).
      Only ``--resolution`` and ``--rho_power`` are used in this case.
    - flag ``-h`` or ``--help`` displays the help message.

    """
    parser = _build_parser()
    args = parser.parse_args()

    if args.n < 0:
        parser.error("the maximum order -n must be a non-negative integer.")
    if args.rho_derivative < 0:
        parser.error("the derivative order -dr must be a non-negative integer.")
    if args.theta_derivative < 0:
        parser.error("the derivative order -dt must be a non-negative integer.")
    if args.resolution < 2:
        parser.error("the resolution must be greater than or equal to 2.")

    if args.interactive:
        display_interactive(resolution=args.resolution, rho_power=args.rho_power)
        return

    if args.radial:
        n = [n_i for n_i in range(args.n + 1) for _ in range(n_i % 2, n_i + 1, 2)]
        m = [m_i for n_i in range(args.n + 1) for m_i in range(n_i % 2, n_i + 1, 2)]
        radial_display(n=n, m=m, rho_derivative=args.rho_derivative, resolution=args.resolution, rho_power=args.rho_power)
    else:
        n, m = zernike_index_to_order(range(zernike_polynomials_count(args.n)))
        zernike_display(n=n, m=m, rho_derivative=args.rho_derivative, theta_derivative=args.theta_derivative, resolution=args.resolution, rho_power=args.rho_power)


def __main_gui__() -> None:
    r"""
    Graphical user interface entry point of the package.

    This method contains the script to run if the user enters the name of the package on the command line
    with the ``gui`` extension. It launches the interactive display window (see :func:`pyzernike.display_interactive`).

    .. code-block:: console

        pyzernike-gui

    """
    display_interactive()


if __name__ == "__main__":
    __main__()