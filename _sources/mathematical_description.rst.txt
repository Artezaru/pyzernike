Mathematical description
===============================

This section describes the mathematics of the Zernike polynomials and the methods used by ``pyzernike`` to compute
them, to extend them to an elliptic annulus, to manipulate their coefficients and to fit them to data:

- :doc:`mathematical_description/conventions`: definition, indexing (OSA/ANSI, Noll, Fringe) and normalization.
- :doc:`mathematical_description/polynomials`: explicit sum, derivatives and recurrence relation.
- :doc:`mathematical_description/elliptic_annulus`: extension to an elliptic annulus and Cartesian derivatives.
- :doc:`mathematical_description/coefficients`: evaluation, conversion, rotation, change of pupil and statistics.
- :doc:`mathematical_description/fitting`: linear and non-linear least squares fits.

.. toctree::
   :maxdepth: 2
   :hidden:

   mathematical_description/conventions
   mathematical_description/polynomials
   mathematical_description/elliptic_annulus
   mathematical_description/coefficients
   mathematical_description/fitting
   mathematical_description/references

Notations
---------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Notation
     - Meaning
   * - :math:`(\rho, \theta)`
     - Polar coordinates on the unit disk, :math:`0 \leq \rho \leq 1`.
   * - :math:`n, m`
     - Radial order :math:`n \geq 0` and azimuthal frequency :math:`m`, with :math:`|m| \leq n` and :math:`n - m` even.
   * - :math:`R_{n}^{m}(\rho)`
     - Radial polynomial (:math:`m \geq 0`).
   * - :math:`Z_{n}^{m}(\rho, \theta)`, :math:`Z_j`
     - Zernike polynomial, of index :math:`j` in a given ordering (OSA/ANSI by default).
   * - :math:`N_{n}^{m}`, :math:`N_j`
     - Normalization factor of Noll.
   * - :math:`N`, :math:`P`
     - Maximum radial order and number :math:`P = \frac{(N+1)(N+2)}{2}` of polynomials with :math:`n \leq N`.
   * - :math:`(R_x, R_y, x_0, y_0, \alpha, h)`
     - Parameters of the elliptic annulus.
   * - :math:`(\rho_{eq}, \theta_{eq})`
     - Equivalent polar coordinates of a point :math:`(x, y)` of the elliptic annulus.
   * - :math:`Zxy_{n}^{m}(x, y)`
     - Zernike polynomial on the elliptic annulus.
   * - :math:`c_j`
     - Coefficients of a sum of Zernike polynomials :math:`f = \sum_j c_j Z_j`.