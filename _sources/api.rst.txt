API Reference
==============

The package ``pyzernike`` is composed of the following functions, classes, and modules.
To learn how to use the package effectively, refer to the documentation :doc:`../usage`.

Computation
----------------

- ``pyzernike.radial_polynomial`` alias ``R`` function is used to compute the radial polynomial of a Zernike polynomial.
- ``pyzernike.zernike_polynomial`` alias ``Z`` function is used to compute the Zernike polynomial.

.. toctree::
   :maxdepth: 1
   :caption: Computation API:

   ./api_doc/core_polynomial
   ./api_doc/radial_polynomial
   ./api_doc/zernike_polynomial

Symbolic Expressions
---------------------

- ``pyzernike.radial_symbolic`` function is used to compute the symbolic radial polynomial.
- ``pyzernike.zernike_symbolic`` function is used to compute the symbolic Zernike polynomial.

.. toctree::
   :maxdepth: 1
   :caption: Symbolic API:

   ./api_doc/core_symbolic
   ./api_doc/radial_symbolic
   ./api_doc/zernike_symbolic


Display
---------------------

- ``pyzernike.radial_display`` function is used to plot the radial polynomial.
- ``pyzernike.zernike_display`` function is used to plot the Zernike polynomial.

.. toctree::
   :maxdepth: 1
   :caption: Display API:

   ./api_doc/core_display
   ./api_doc/radial_display
   ./api_doc/zernike_display


Additional Functions
---------------------

- ``pyzernike.zernike_index_to_order`` function is used to convert Zernike indices to their corresponding orders (n, m).
- ``pyzernike.zernike_order_to_index`` function is used to convert Zernike orders (n, m) to their corresponding indices.
- ``pyzernike.zernike_polynomial_up_to_order`` alias ``Zup`` function is used to compute the Zernike polynomial up to a specified order.
- ``pyzernike.xy_zernike_polynomial`` alias ``Zxy`` function is used to compute the Zernike polynomial in Cartesian coordinates (x, y) in an extended domain ``G``.
- ``pyzernike.xy_zernike_polynomial_up_to_order`` alias ``Zxyup`` function is used to compute the Zernike polynomial in Cartesian coordinates (x, y) in an extended domain ``G`` up to a specified order.

.. toctree::
   :maxdepth: 1
   :caption: Additional Functions API:

   ./api_doc/zernike_index_to_order
   ./api_doc/zernike_order_to_index
   ./api_doc/zernike_polynomial_up_to_order
   ./api_doc/xy_zernike_polynomial
   ./api_doc/xy_zernike_polynomial_up_to_order


