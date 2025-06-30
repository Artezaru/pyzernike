Usage
==============

The package ``pyzernike`` is a Python package to compute Zernike polynomials and their derivatives.

To compute the Zernike polynomials :math:`Z_{n}^{m}`, use the following code:

.. code-block:: python

    from pyzernike import zernike_polynomial
    import numpy as np

    rho = np.linspace(0, 1, 100)
    theta = np.linspace(0, 2*np.pi, 100)

    n = 3
    m = 1
    result = zernike_polynomial(rho, theta, [n], [m])
    Z_31 = result[0] # result is a list of Zernike polynomials for given n and m

To compute the second derivatives of the Zernike polynomials :math:`Z_{n,m}` with respect to :math:`\rho`:

.. code-block:: python

    from pyzernike import zernike_polynomial
    import numpy as np

    rho = np.linspace(0, 1, 100)
    theta = np.linspace(0, 2*np.pi, 100)

    n = 3
    m = 1
    Z_31_drho_drho = zernike_polynomial(rho, theta, [n], [m], rho_derivative=[2])[0]


To compute several Zernike polynomials at once, you can pass lists of :math:`n`, :math:`m`, and their derivatives:

.. code-block:: python

    from pyzernike import zernike_polynomial
    import numpy as np

    rho = np.linspace(0, 1, 100)
    theta = np.linspace(0, 2*np.pi, 100)

    n = [3, 4, 5]
    m = [1, 2, 3]
    dr = [2, 1, 0]  # Derivatives with respect to rho for each Zernike polynomial
    theta_derivative = [0, 1, 2]  # Derivatives with respect to theta for each Zernike polynomial

    result = zernike_polynomial(rho, theta, n, m, rho_derivative=dr, theta_derivative=theta_derivative)

    Z_31_drho_drho = result[0]  # Zernike polynomial for n=3, m=1 with second derivative with respect to rho
    Z_42_drho_dtheta = result[1]  # Zernike polynomial for n=4, m=2 with first derivative with respect to theta and first derivative with respect to rho
    Z_53_dtheta_dtheta = result[2]  # Zernike polynomial for n=5, m=3 with second derivative with respect to theta
