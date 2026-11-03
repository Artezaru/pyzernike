Definition and conventions
==========================

.. contents:: Table of Contents
   :local:
   :depth: 2

Definition of the Zernike polynomials
-------------------------------------

The Zernike polynomials [Zernike1934]_ :math:`Z_{n}^{m}(\rho, \theta)` are defined on the unit disk
:math:`(0 \leq \rho \leq 1, \theta)` for a radial order :math:`n \geq 0` and an azimuthal frequency :math:`m` with
:math:`|m| \leq n` and :math:`n - m` even:

.. math::

    Z_{n}^{m}(\rho, \theta) = \begin{cases}
        R_{n}^{m}(\rho) \cos(m \theta) & \text{if } m \geq 0 \\
        R_{n}^{|m|}(\rho) \sin(|m| \theta) & \text{if } m < 0
    \end{cases}

where :math:`R_{n}^{m}` is the radial polynomial (:math:`m \geq 0`):

.. math::

    R_{n}^{m}(\rho) = \sum_{k=0}^{(n-m)/2} \frac{(-1)^k (n-k)!}{k! \left(\frac{n+m}{2} - k\right)! \left(\frac{n-m}{2} - k\right)!} \rho^{n-2k}

The radial polynomials satisfy :math:`R_{n}^{m}(1) = 1`, and :math:`R_{n}^{m}(\rho) = \rho^{m} P(\rho^2)` with
:math:`P` a polynomial of degree :math:`\frac{n-m}{2}`. The Zernike polynomials are therefore polynomials of degree
:math:`n` in the Cartesian coordinates :math:`(\rho \cos\theta, \rho \sin\theta)`.

.. note::

    If :math:`|m| > n` or :math:`n - m` is odd, the polynomial is set to zero. A negative order :math:`n` is an error.

The sign convention for :math:`m < 0` (the sine term is :math:`+\sin(|m|\theta)`) is the one of the OSA/ANSI
standard Z80.28 [Thibos2002]_. Some authors use :math:`-\sin(m\theta)` with :math:`m < 0`, which is the same function,
or associate the sine with :math:`m > 0`, which changes the sign or the role of some polynomials: the conventions of
a set of coefficients must always be checked.

Indexing conventions
--------------------

A single index :math:`j` is often used instead of the pair :math:`(n, m)`. Three orderings are available with the
argument ``indexing`` of :func:`pyzernike.zernike_index_to_order` and :func:`pyzernike.zernike_order_to_index`, and
all the functions working on vectors of coefficients use the OSA/ANSI ordering.

**OSA/ANSI** (first index 0) [Thibos2002]_: the polynomials are sorted by radial order :math:`n`, then by azimuthal
frequency :math:`m` from :math:`-n` to :math:`n`:

.. math::

    j = \frac{n(n + 2) + m}{2}

**Noll** (first index 1) [Noll1976]_: the polynomials are sorted by radial order :math:`n`, then by :math:`|m|`. The
even indices are given to the cosine terms (:math:`m > 0`) and the odd indices to the sine terms (:math:`m < 0`):

.. math::

    j = \frac{n(n + 1)}{2} + |m| + s
    \quad \text{with} \quad
    s = \begin{cases}
        0 & \text{if } m > 0 \text{ and } n \bmod 4 \in \{0, 1\}, \text{ or } m < 0 \text{ and } n \bmod 4 \in \{2, 3\} \\
        1 & \text{otherwise}
    \end{cases}

**Fringe** (first index 1), also called University of Arizona ordering: the polynomials are sorted by
:math:`\frac{n + |m|}{2}`, then by decreasing :math:`|m|`, the cosine term before the sine term:

.. math::

    j = \left(1 + \frac{n + |m|}{2}\right)^2 - 2|m| + \begin{cases} 1 & \text{if } m < 0 \\ 0 & \text{otherwise} \end{cases}

The OSA/ANSI and Noll orderings contain exactly all the polynomials up to the order :math:`N` in their first
:math:`P = \frac{(N+1)(N+2)}{2}` indices. The Fringe ordering is not sorted by radial order: for example,
:math:`Z_{4}^{0}` (index 9) comes before :math:`Z_{3}^{3}` (index 10).

.. list-table:: The first polynomials in the three orderings
   :header-rows: 1
   :widths: 14 14 14 14 44

   * - :math:`(n, m)`
     - OSA/ANSI
     - Noll
     - Fringe
     - Name
   * - :math:`(0, 0)`
     - 0
     - 1
     - 1
     - Piston
   * - :math:`(1, -1)`
     - 1
     - 3
     - 3
     - Tilt (vertical)
   * - :math:`(1, 1)`
     - 2
     - 2
     - 2
     - Tilt (horizontal)
   * - :math:`(2, -2)`
     - 3
     - 5
     - 6
     - Oblique astigmatism
   * - :math:`(2, 0)`
     - 4
     - 4
     - 4
     - Defocus
   * - :math:`(2, 2)`
     - 5
     - 6
     - 5
     - Vertical astigmatism
   * - :math:`(3, -3)`
     - 6
     - 9
     - 11
     - Vertical trefoil
   * - :math:`(3, -1)`
     - 7
     - 7
     - 8
     - Vertical coma
   * - :math:`(3, 1)`
     - 8
     - 8
     - 7
     - Horizontal coma
   * - :math:`(3, 3)`
     - 9
     - 10
     - 10
     - Oblique trefoil
   * - :math:`(4, 0)`
     - 12
     - 11
     - 9
     - Primary spherical aberration

Vectors of coefficients are converted between the orderings with :func:`pyzernike.zernike_coefficients_convert`
(see :doc:`coefficients`).

Normalization and orthogonality
-------------------------------

The Zernike polynomials are orthogonal on the unit disk. With the inner product averaged over the disk:

.. math::

    \langle f, g \rangle = \frac{1}{\pi} \int_{0}^{2\pi} \int_{0}^{1} f(\rho, \theta) \, g(\rho, \theta) \, \rho \, d\rho \, d\theta

they satisfy:

.. math::

    \langle Z_{n}^{m}, Z_{n'}^{m'} \rangle = \frac{\delta_{nn'} \delta_{mm'}}{\left(N_{n}^{m}\right)^2}
    \quad \text{with} \quad
    N_{n}^{m} = \sqrt{(2 - \delta_{m0})(n + 1)} = \begin{cases}
        \sqrt{n + 1} & \text{if } m = 0 \\
        \sqrt{2(n + 1)} & \text{if } m \neq 0
    \end{cases}

The factors :math:`N_{n}^{m}` are the normalization factors of Noll [Noll1976]_, given by
:func:`pyzernike.zernike_normalization`. The normalized polynomials :math:`N_{n}^{m} Z_{n}^{m}` are orthonormal, and
are computed by all the functions with the argument ``normalize=True``. By default (``normalize=False``), the
polynomials are not normalized, so that :math:`R_{n}^{m}(1) = 1`.

The normalization has two practical consequences for a function :math:`f = \sum_j c_j Z_j`:

- The coefficients depend on the normalization: :math:`c_j Z_j = c'_j (N_j Z_j)`, so :math:`c'_j = c_j / N_j`.
- Only the piston :math:`Z_0^0 = 1` has a non-zero mean, so the variance of :math:`f` on the disk is
  :math:`\sigma^2 = \sum_{j > 0} c_j^2 / N_j^2`, i.e. :math:`\sum_{j > 0} c_j^2` for the normalized polynomials
  (see :func:`pyzernike.zernike_rms`).

.. note::

    The normalization factors are defined for the full Zernike polynomials only: the radial polynomials
    (:func:`pyzernike.radial_polynomial`) are never normalized.