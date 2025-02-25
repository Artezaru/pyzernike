Usage
==============

The package ``pyzernike`` is a Python package to compute Zernike polynomials and their derivatives.

To compute the Zernike polynomials :math:`Z_{n,m}`, you can use the following functions:

.. code-block:: python

    from pyzernike import zernike_polynomial
    import numpy as np

    rho = np.linspace(0, 1, 100)
    theta = np.linspace(0, 2*np.pi, 100)

    n = 3
    m = 1
    Z_31 = zernike_polynomial(rho, theta, n, m)

To compute the second derivatives of the Zernike polynomials :math:`Z_{n,m}` with respect to :math:`\rho`:

.. code-block:: python

    from pyzernike import zernike_derivative
    import numpy as np

    rho = np.linspace(0, 1, 100)
    theta = np.linspace(0, 2*np.pi, 100)

    n = 3
    m = 1
    Z_31_drho_drho = zernike_derivative(rho, theta, n, m, rho_derivative=2)


