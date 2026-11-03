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

from pyzernike import display_interactive, xy_display_interactive


def _build_parser() -> argparse.ArgumentParser:
    r"""
    Build the parser of the command line arguments.

    Returns
    -------
    argparse.ArgumentParser
        The parser, with the only flag ``--xy``.

    """
    parser = argparse.ArgumentParser(prog="pyzernike", description="Explore the Zernike polynomials in an interactive window.")
    parser.add_argument(
        '--xy', action='store_true',
        help="Explore the Zernike polynomials on an elliptic annulus in Cartesian coordinates (x, y) instead of the unit disk."
    )
    return parser


def __main__() -> None:
    r"""
    Main entry point of the package.

    This method contains the script to run if the user enters the name of the package on the command line.
    It launches the interactive display window of the Zernike polynomials on the unit disk
    (see :func:`pyzernike.display_interactive`), or on an elliptic annulus with the flag ``--xy``
    (see :func:`pyzernike.xy_display_interactive`). All the parameters (orders, derivatives, resolution,
    parameters of the annulus) are set in the window.

    .. code-block:: console

        pyzernike
        pyzernike --xy

    - flag ``--xy`` explores the Zernike polynomials on an elliptic annulus in Cartesian coordinates :math:`(x, y)`.
    - flag ``-h`` or ``--help`` displays the help message.

    """
    args = _build_parser().parse_args()

    if args.xy:
        xy_display_interactive()
    else:
        display_interactive()


def __main_gui__() -> None:
    r"""
    Graphical user interface entry point of the package, kept for compatibility.

    This method contains the script to run if the user enters the name of the package on the command line
    with the ``gui`` extension. It is identical to the main entry point (see :func:`__main__`).

    .. code-block:: console

        pyzernike-gui
        pyzernike-gui --xy

    """
    __main__()


if __name__ == "__main__":
    __main__()