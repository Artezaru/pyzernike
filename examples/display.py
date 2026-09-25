try:
    import pyzernike
except:
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import pyzernike

from pyzernike import zernike_display, radial_display, display_interactive

zernike_display(n=[2, 3, 4], m=[0, 1, -2])
radial_display(n=[2, 3, 4], m=[0, 1, 2])
display_interactive()