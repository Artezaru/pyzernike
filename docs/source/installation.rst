Installation
============

To install ``pyzernike``, you can use the following command:

.. code-block:: bash

    pip install pyzernike-polynomials

Or install it directly from the GitHub repository using:

.. code-block:: bash

    pip install git+https://github.com/Artezaru/pyzernike.git

The package is imported as ``pyzernike`` (the name ``pyzernike-polynomials`` is only used by ``pip``):

.. code-block:: python

    import pyzernike

    print(pyzernike.__version__)


Optional features
------------------------------

The core of ``pyzernike`` (computation of the polynomials, fit, coefficients, conventions) only requires ``numpy``.
The display and the symbolic expressions require additional packages, which can be installed with the
corresponding extras:

.. code-block:: bash

    pip install "pyzernike-polynomials[display]"    # matplotlib, for the display functions
    pip install "pyzernike-polynomials[symbolic]"   # sympy, for the symbolic expressions
    pip install "pyzernike-polynomials[all]"        # all the optional features

The quotes avoid the interpretation of the brackets by some shells (e.g. ``zsh``).

The extras only install the optional dependencies: ``pyzernike`` does not keep track of how it was installed.
Installing ``matplotlib`` or ``sympy`` later, by any means (``pip install sympy``, ``conda``, ...), enables the
corresponding features as well.

The optional subpackages are imported only when they are used: ``import pyzernike`` never imports ``matplotlib``
or ``sympy``. If an optional dependency is missing, the first use of a function that needs it raises an
``ImportError`` that names the missing package and gives the command to install it:

.. code-block:: python

    import pyzernike

    pyzernike.zernike_polynomial(0.5, 0.0, n=2, m=0)  # works with numpy only
    pyzernike.zernike_symbolic(n=2, m=0)              # ImportError if sympy is not installed

.. code-block:: text

    ImportError: The symbolic functions of pyzernike require the package 'sympy', which is not installed in this
    Python environment. Install it with:

        pip install sympy

    (or install pyzernike with its optional dependencies: pip install "pyzernike-polynomials[symbolic]").
    If you have just installed it, restart the Python session.

.. note::

    In an interactive session (e.g. a Jupyter notebook), restart the session (or the kernel) after installing a
    missing dependency: a package installed while Python is running may not be found until the next start.

Dependencies
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Package
     - Required for
     - Use
   * - ``numpy``
     - always
     - All the numerical computations.
   * - ``matplotlib``
     - ``[display]``
     - :func:`pyzernike.radial_display`, :func:`pyzernike.zernike_display`, :func:`pyzernike.display_interactive`,
       :func:`pyzernike.xy_zernike_display`, :func:`pyzernike.xy_display_interactive` and the command line
       interface (``python -m pyzernike``).
   * - ``sympy``
     - ``[symbolic]``
     - :func:`pyzernike.radial_symbolic` and :func:`pyzernike.zernike_symbolic`.
   * - ``pytest``, ``scipy``
     - ``[dev]``
     - The tests (``scipy`` only for the comparison with ``scipy.optimize.least_squares``).

``scipy`` is not a dependency of ``pyzernike``: the functions returned by :func:`pyzernike.xy_zernike_least_squares`
can be given to any least squares solver, including ``scipy.optimize.least_squares`` if it is installed.