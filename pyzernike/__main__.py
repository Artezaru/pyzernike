from pyzernike import zernike_display

def __main__() -> None:
    r"""
    Main entry point of the package.

    This method contains the script to run if the user enter the name of the package on the command line. 

    .. code-block:: console
        pyzernike
        
    """
    list_n = []
    list_m = []
    for n in range(0, 5):
        for m in range(-n, n + 1, 2):
            list_n.append(n)
            list_m.append(m)
    
    zernike_display(n=list_n, m=list_m)

def __main_gui__() -> None:
    r"""
    Graphical user interface entry point of the package.

    This method contains the script to run if the user enter the name of the package on the command line with the ``gui`` extension.

    .. code-block:: console
        pyzernike-gui
        
    """
    raise NotImplementedError("The graphical user interface entry point is not implemented yet.")

