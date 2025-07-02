from pyzernike import zernike_display, radial_display
import argparse

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

    args = parser.parse_args()

    Nzer = args.n
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
        radial_display(n=list_n, m=list_m)
    else:
        zernike_display(n=list_n, m=list_m)

def __main_gui__() -> None:
    r"""
    Graphical user interface entry point of the package.

    This method contains the script to run if the user enter the name of the package on the command line with the ``gui`` extension.

    .. code-block:: console
        pyzernike-gui
        
    """
    raise NotImplementedError("The graphical user interface entry point is not implemented yet.")

