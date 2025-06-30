Usage
==============

The package ``pyzernike`` is a Python package to compute Zernike polynomials and their derivatives.

Compute Zernike Polynomials
------------------------------

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


.. seealso:: 
    
    - :func:`pyzernike.zernike_polynomial` for more details on the function parameters and usage.
    - :func:`pyzernike.radial_polynomial` for computing radial polynomials.
    - :func:`pyzernike.xy_zernike_polynomial` for cartesian extended Zernike polynomials.


Get the mathematical expression of Zernike Polynomials
------------------------------------------------------

To get the mathematical expression of Zernike polynomials, you can use the `zernike_symbolic` function:

.. code-block:: python

    from pyzernike import zernike_symbolic

    n = 3
    m = 1
    expression = zernike_symbolic([n], [m])
    print(expression[0])  # This will print the symbolic expression of Zernike polynomial Z_31

.. note::

    ``x`` is the symbol for :math:`\rho` in the symbolic expression, and ``y`` is the symbol for :math:`\theta`. 
    You can use these symbols to manipulate the expressions further if needed.

.. code-block:: python

    import numpy
    import sympy
    rho = numpy.linspace(0, 1, 100)
    theta = numpy.linspace(0, 2 * numpy.pi, 100)

    # `x` represents the radial coordinate in the symbolic expression
    # `y` represents the angular coordinate in the symbolic expression
    
    func = sympy.lambdify(['x', 'y'], expression, 'numpy')
    evaluated_result = func(rho, theta)

.. seealso:: 

    - :func:`pyzernike.zernike_symbolic` for more details on the function parameters and usage.
    - :func:`pyzernike.radial_symbolic` for computing symbolic radial polynomials.

Display Zernike Polynomials
-----------------------------

To visualize the Zernike polynomials, you can use the `zernike_display` function. This function generates plots for the specified Zernike polynomials.

.. code-block:: python

    from pyzernike import zernike_display

    n = [0, 1, 2, 3, 4]
    m = [0, 1, -1, 2, -2]
    zernike_display(n=n, m=m)

.. image:: ../../pyzernike/resources/zernike_display.png
    :align: center
    :width: 600px

.. seealso::

    - :func:`pyzernike.zernike_display` for more details on the function parameters and usage.
    - :func:`pyzernike.radial_display` for displaying radial Zernike polynomials.