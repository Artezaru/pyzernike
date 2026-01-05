API Reference
==============

The package ``pyzernike`` is composed of the following functions, classes, and modules.
To learn how to use the package effectively, refer to the documentation :doc:`../usage`.

.. contents:: Table of Contents
   :local:
   :depth: 1
   :backlinks: top

Generalities and notations about Zernike Polynomials
-------------------------------------------------------

For a polynomial of order/degree :math:`n` and azimuthal frequency :math:`m`, the Zernike polynomial :math:`Z_{n}^{m}(\rho, \theta)` is defined on the unit disk as:

.. math::

      Z_{n}^{m}(\rho, \theta) = R_{n}^{m}(\rho) \cos(m \theta) \quad \text{if} \quad m \geq 0
   
.. math::

   Z_{n}^{m}(\rho, \theta) = R_{n}^{-m}(\rho) \sin(-m \theta) \quad \text{if} \quad m < 0

Where :math:`R_{n}^{m}(\rho)` is the radial polynomial defined as:

.. math::

   R_{n}^{m}(\rho) = \sum_{k=0}^{(n-m)/2} \frac{(-1)^k (n-k)!}{k! ((n+m)/2 - k)! ((n-m)/2 - k)!} \rho^{n-2k}

The derivative of order (derivative (a)) of the Zernike polynomial with respect to rho and order (derivative (b)) with respect to theta is defined as follows :

.. math::

   \frac{\partial^{a}\partial^{b}Z_{n}^{m}(\rho, \theta)}{\partial \rho^{a} \partial \theta^{b}} = \frac{\partial^{a}R_{n}^{m}(\rho)}{\partial \rho^{a}} \frac{\partial^{b}\cos(m \theta)}{\partial \theta^{b}} \quad \text{if} \quad m > 0

.. note::

   Thus any polynomial evaluated at :math:`(\rho, \theta)` can be defined by 4 parameters :

   - n : order/degree of the polynomial noted ``n`` in the documentation
   - m : azimuthal frequency of the polynomial noted ``m`` in the documentation
   - a : order of the derivative with respect to :math:`\rho` noted ``rho_derivative`` in the arguments.
   - b : order of the derivative with respect to :math:`\theta` noted ``theta_derivative`` in the arguments.


Computation of the Polynomials
--------------------------------

- ``radial_polynomial`` computes radial polynomials for several sets of ``[n, m, a]`` at once.
- ``zernike_polynomial`` computes Zernike polynomials for several sets of ``[n, m, a, b]`` at once.
- ``zernike_polynomial_up_to_order`` computes all Zernike polynomials up to a specified order for common sets of ``[a, b]``.
- ``xy_zernike_polynomial`` computes the Zernike polynomial in Cartesian coordinates (x, y) in an extended elliptic annulus domain ``G``.

.. toctree::
   :maxdepth: 1
   :caption: Computation API:

   ./api_doc/radial_polynomial
   ./api_doc/zernike_polynomial
   ./api_doc/zernike_polynomial_up_to_order
   ./api_doc/xy_zernike_polynomial

Symbolic Expressions
---------------------

- ``radial_symbolic`` obtains the symbolic ``sympy`` radial polynomials for several sets of ``[n, m, a]`` at once.
- ``zernike_symbolic`` obtains the symbolic ``sympy`` Zernike polynomials for several sets of ``[n, m, a, b]`` at once.

.. toctree::
   :maxdepth: 1
   :caption: Symbolic API:

   ./api_doc/radial_symbolic
   ./api_doc/zernike_symbolic


Display
---------------------

- ``radial_display`` displays the radial polynomials for several sets of ``[n, m, a]`` at once in a interactive plot.
- ``zernike_display`` displays the Zernike polynomials for several sets of ``[n, m, a, b]`` at once in a interactive plot.

.. toctree::
   :maxdepth: 1
   :caption: Display API:

   ./api_doc/radial_display
   ./api_doc/zernike_display


Additional Functions
---------------------

- ``zernike_index_to_order`` converts Zernike OSA/ANSI indices to their corresponding orders (n, m).
- ``zernike_order_to_index`` converts Zernike orders (n, m) to their corresponding OSA/ANSI indices.
- ``cartesian_to_elliptic_annulus`` converts Cartesian coordinates (x, y) to elliptic annulus coordinates (rho, theta) in an extended elliptic annulus domain ``G``.
- ``xy_zernike_polynomial_up_to_order`` computes all the Zernike polynomials up to a specified order in Cartesian coordinates (x, y) in an extended elliptic annulus domain ``G``.

.. toctree::
   :maxdepth: 1
   :caption: Additional Functions API:

   ./api_doc/zernike_index_to_order
   ./api_doc/zernike_order_to_index
   ./api_doc/cartesian_to_elliptic_annulus
   ./api_doc/xy_zernike_polynomial_up_to_order


Core Development
---------------------

For developers interested in the core functionalities of the package, the ``core`` module provides essential functions that underpin the computations and symbolic representations of Zernike polynomials.

.. toctree::
   :maxdepth: 1
   :caption: Core Development API:

   ./api_doc/core/core_polynomial
   ./api_doc/core/core_symbolic
   ./api_doc/core/core_display
   ./api_doc/core/core_create_precomputing_terms
   ./api_doc/core/core_corresponding_signed_integer_type
   ./api_doc/core/core_cartesian_to_elliptic_annulus
