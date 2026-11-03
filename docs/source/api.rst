API Reference
==============

All the functions are available at the top level of the package (``import pyzernike`` then
``pyzernike.zernike_polynomial(...)``). To learn how to use them, see :doc:`usage`; for the definitions and the
methods, see :doc:`mathematical_description`.

.. contents:: Table of Contents
   :local:
   :depth: 1
   :backlinks: top

Common arguments
----------------

The functions share the same names of arguments and the same conventions:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Argument
     - Meaning
   * - ``rho``, ``theta``
     - Polar coordinates on the unit disk (:math:`0 \leq \rho \leq 1`, :math:`\theta` in radians).
   * - ``x``, ``y``
     - Cartesian coordinates, for the functions prefixed by ``xy_``.
   * - ``n``, ``m``
     - Radial order :math:`n` and azimuthal frequency :math:`m` of a polynomial :math:`Z_{n}^{m}`
       (:math:`|m| \leq n`, :math:`n - m` even). See :doc:`mathematical_description/conventions`.
   * - ``order``
     - Maximum radial order :math:`N`: all the :math:`P = \frac{(N+1)(N+2)}{2}` polynomials with
       :math:`n \leq N` are used, in the OSA/ANSI ordering.
   * - ``rho_derivative``, ``theta_derivative``
     - Orders :math:`a` and :math:`b` of the derivatives with respect to :math:`\rho` and :math:`\theta`.
   * - ``x_derivative``, ``y_derivative``
     - Orders :math:`d_x` and :math:`d_y` of the derivatives with respect to :math:`x` and :math:`y`.
   * - ``Rx``, ``Ry``, ``x0``, ``y0``, ``alpha``, ``h``
     - Parameters of the elliptic annulus: semi-axes, center, rotation angle (radians) and ratio
       :math:`h \in [0, 1)` of the inner to the outer semi-axes. The default is the unit disk. See
       :doc:`mathematical_description/elliptic_annulus`.
   * - ``coefficients``
     - Coefficients :math:`c_j` of a sum :math:`\sum_j c_j Z_j`, in the OSA/ANSI ordering, for all the polynomials
       up to an order (1, 3, 6, 10, 15, ... values). See :doc:`mathematical_description/coefficients`.
   * - ``normalize``
     - If True, the polynomials (or the coefficients) are those normalized by the factors of Noll.
   * - ``use_recursive``
     - If True, the polynomials are computed with the recurrence relation (recommended for high orders).
   * - ``default``
     - Value set outside the domain (``numpy.nan`` by default).
   * - ``to_list``
     - Output as a list of arrays (True) or as a single array (False).

Single integers and sequences of integers can be mixed for ``n``, ``m`` and the derivative orders: the integers
are repeated to the length of the sequences.

Polynomials on the unit disk
----------------------------

.. list-table::
   :widths: 40 60

   * - :func:`pyzernike.zernike_polynomial`
     - Zernike polynomials :math:`Z_{n}^{m}(\rho, \theta)` and their derivatives, for given :math:`(n, m)`.
   * - :func:`pyzernike.zernike_polynomial_up_to_order`
     - All the Zernike polynomials up to an order and their derivatives.
   * - :func:`pyzernike.radial_polynomial`
     - Radial polynomials :math:`R_{n}^{m}(\rho)` and their derivatives.

.. toctree::
   :maxdepth: 1
   :caption: Polynomials on the unit disk
   :hidden:

   ./api_doc/zernike_polynomial
   ./api_doc/zernike_polynomial_up_to_order
   ./api_doc/radial_polynomial

Polynomials on an elliptic annulus
----------------------------------

.. list-table::
   :widths: 40 60

   * - :func:`pyzernike.xy_zernike_polynomial`
     - Zernike polynomials in Cartesian coordinates on an elliptic annulus, and their derivatives of any order with
       respect to :math:`x` and :math:`y`.
   * - :func:`pyzernike.xy_zernike_polynomial_up_to_order`
     - All the Zernike polynomials up to an order on an elliptic annulus, and their derivatives.
   * - :func:`pyzernike.xy_zernike_parameter_derivative`
     - Derivatives of the polynomials with respect to the parameters :math:`(x_0, y_0, R_x, R_y, \alpha, h)` of the
       annulus.
   * - :func:`pyzernike.cartesian_to_elliptic_annulus`
     - Equivalent polar coordinates :math:`(\rho_{eq}, \theta_{eq})` of Cartesian points, and their derivatives.

.. toctree::
   :maxdepth: 1
   :caption: Polynomials on an elliptic annulus
   :hidden:

   ./api_doc/xy_zernike_polynomial
   ./api_doc/xy_zernike_polynomial_up_to_order
   ./api_doc/xy_zernike_parameter_derivative
   ./api_doc/cartesian_to_elliptic_annulus

Indexing and normalization
--------------------------

.. list-table::
   :widths: 40 60

   * - :func:`pyzernike.zernike_index_to_order`
     - Orders :math:`(n, m)` of indices in the OSA/ANSI, Noll or Fringe convention.
   * - :func:`pyzernike.zernike_order_to_index`
     - Indices of orders :math:`(n, m)` in the OSA/ANSI, Noll or Fringe convention.
   * - :func:`pyzernike.zernike_polynomials_count`
     - Number of polynomials up to an order.
   * - :func:`pyzernike.zernike_normalization`
     - Normalization factors of Noll.

.. toctree::
   :maxdepth: 1
   :caption: Indexing and normalization
   :hidden:

   ./api_doc/zernike_index_to_order
   ./api_doc/zernike_order_to_index
   ./api_doc/zernike_polynomials_count
   ./api_doc/zernike_normalization

Coefficients
------------

.. list-table::
   :widths: 40 60

   * - :func:`pyzernike.zernike_evaluate`
     - Sum of polynomials :math:`\sum_j c_j Z_j` and its derivatives on the unit disk.
   * - :func:`pyzernike.xy_zernike_evaluate`
     - Sum of polynomials and its derivatives on an elliptic annulus.
   * - :func:`pyzernike.zernike_coefficients_convert`
     - Conversion of coefficients between the indexing conventions and the normalizations.
   * - :func:`pyzernike.zernike_coefficients_rotate`
     - Coefficients in a rotated pupil.
   * - :func:`pyzernike.zernike_coefficients_transform`
     - Coefficients in a scaled, decentered and rotated pupil.
   * - :func:`pyzernike.zernike_rms`
     - Root mean square of a sum of polynomials.
   * - :func:`pyzernike.zernike_peak_to_valley`
     - Peak-to-valley of a sum of polynomials.
   * - :func:`pyzernike.zernike_strehl_ratio`
     - Strehl ratio of a wavefront (extended Maréchal approximation).

.. toctree::
   :maxdepth: 1
   :caption: Coefficients
   :hidden:

   ./api_doc/zernike_evaluate
   ./api_doc/xy_zernike_evaluate
   ./api_doc/zernike_coefficients_convert
   ./api_doc/zernike_coefficients_rotate
   ./api_doc/zernike_coefficients_transform
   ./api_doc/zernike_rms
   ./api_doc/zernike_peak_to_valley
   ./api_doc/zernike_strehl_ratio

Fitting
-------

.. list-table::
   :widths: 40 60

   * - :func:`pyzernike.zernike_fit`
     - Least squares fit of sampled data on the unit disk.
   * - :func:`pyzernike.xy_zernike_fit`
     - Least squares fit of sampled data on an elliptic annulus.
   * - :func:`pyzernike.xy_zernike_least_squares`
     - Residual and Jacobian functions to estimate the parameters of the annulus with any least squares solver.

.. toctree::
   :maxdepth: 1
   :caption: Fitting
   :hidden:

   ./api_doc/zernike_fit
   ./api_doc/xy_zernike_fit
   ./api_doc/xy_zernike_least_squares

Symbolic expressions
--------------------

These functions require ``sympy`` (``pip install sympy``, see :doc:`installation`).

.. list-table::
   :widths: 40 60

   * - :func:`pyzernike.zernike_symbolic`
     - Exact ``sympy`` expressions of the Zernike polynomials and of their derivatives.
   * - :func:`pyzernike.radial_symbolic`
     - Exact ``sympy`` expressions of the radial polynomials and of their derivatives.

.. toctree::
   :maxdepth: 1
   :caption: Symbolic expressions
   :hidden:

   ./api_doc/zernike_symbolic
   ./api_doc/radial_symbolic

Display
-------

These functions require ``matplotlib`` (``pip install matplotlib``, see :doc:`installation`).

.. list-table::
   :widths: 40 60

   * - :func:`pyzernike.zernike_display`
     - Figure of given Zernike polynomials on the unit disk.
   * - :func:`pyzernike.radial_display`
     - Figure of given radial polynomials.
   * - :func:`pyzernike.display_interactive`
     - Interactive window to explore the polynomials on the unit disk (command ``pyzernike``).
   * - :func:`pyzernike.xy_zernike_display`
     - Figure of given Zernike polynomials on an elliptic annulus.
   * - :func:`pyzernike.xy_display_interactive`
     - Interactive window to explore the polynomials on an elliptic annulus (command ``pyzernike --xy``).

.. toctree::
   :maxdepth: 1
   :caption: Display
   :hidden:

   ./api_doc/zernike_display
   ./api_doc/radial_display
   ./api_doc/display_interactive
   ./api_doc/xy_zernike_display
   ./api_doc/xy_display_interactive