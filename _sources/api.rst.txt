API Reference
==============

The package ``pyzernike`` is composed of the following functions, classes, and modules:

- ``pyzernike.radial_polynomial`` alias ``R`` function is used to compute the radial polynomial of a Zernike polynomial.
- ``pyzernike.radial_symbolic`` function is used to compute the symbolic radial polynomial.
- ``pyzernike.radial_display`` function is used to plot the radial Zernike polynomials.
- ``pyzernike.zernike_polynomial`` alias ``Z`` function is used to compute the Zernike polynomial.
- ``pyzernike.zernike_symbolic`` function is used to compute the symbolic Zernike polynomial.
- ``pyzernike.zernike_display`` function is used to plot the Zernike polynomials.
- ``pyzernike.xy_zernike_polynomial`` alias ``XYZ`` function

.. toctree::
   :maxdepth: 1
   :caption: API:

   ./api_doc/core_polynomial
   ./api_doc/core_symbolic
   ./api_doc/core_display
   ./api_doc/radial_polynomial
   ./api_doc/radial_symbolic
   ./api_doc/radial_display
   ./api_doc/zernike_polynomial
   ./api_doc/zernike_symbolic
   ./api_doc/zernike_display
   ./api_doc/xy_zernike_polynomial

Some deprecated functions are still available in the package ``pyzernike.deprecated``, but they are not recommended for use in new code. They may be removed in future versions of the package. It is advisable to use the newer functions and classes that have been introduced.
They are not documented in this API reference, but you can find them in the source code of the package.

To learn how to use the package effectively, refer to the documentation :doc:`../usage`.