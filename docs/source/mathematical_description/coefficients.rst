Coefficients of a sum of polynomials
====================================

.. contents:: Table of Contents
   :local:
   :depth: 2

A function on the unit disk (a wavefront, the sag of a surface, a distortion field, ...) is usually described by the
coefficients :math:`c_j` of its expansion on the Zernike polynomials:

.. math::

    f(\rho, \theta) = \sum_{j=0}^{P-1} c_j Z_j(\rho, \theta)

In ``pyzernike``, a vector of coefficients contains all the polynomials up to a maximum radial order :math:`N` in the
OSA/ANSI ordering, so its length is :math:`P = \frac{(N+1)(N+2)}{2}` (1, 3, 6, 10, 15, ...). The coefficients of the
polynomials not used are zero. This page describes the operations on these vectors.

Evaluation
----------

The function :math:`f` and its derivatives are evaluated by :func:`pyzernike.zernike_evaluate` (polar coordinates)
and :func:`pyzernike.xy_zernike_evaluate` (elliptic annulus, derivatives with respect to :math:`x` and :math:`y`):

.. math::

    \frac{\partial^{a+b} f}{\partial \rho^{a} \partial \theta^{b}} = \sum_{j} c_j \frac{\partial^{a+b} Z_j}{\partial \rho^{a} \partial \theta^{b}}

The derivatives of a sum are the sums of the derivatives, so they are computed with the methods of
:doc:`polynomials` and :doc:`elliptic_annulus` (explicit sums or recurrence relation). The :math:`P` polynomials are
computed by chunks of points (at most about :math:`2^{22}` values at once), so that the memory does not grow with
:math:`P` times the number of points.

Conversion between conventions
------------------------------

:func:`pyzernike.zernike_coefficients_convert` converts a vector of coefficients between the indexing conventions
and the normalizations of :doc:`conventions`, without changing the function :math:`f`:

- **Indexing.** The coefficient of each polynomial :math:`Z_{n}^{m}` is moved from its index in one ordering to its
  index in the other. The coefficients of the polynomials missing in the input are set to zero. An output in the
  OSA/ANSI or Noll ordering contains all the polynomials up to the maximum order of the input; an output in the
  Fringe ordering ends at the largest Fringe index of the input, as this ordering is not sorted by radial order.
- **Normalization.** From :math:`c_j Z_j = c'_j (N_j Z_j)`:

  .. math::

      c'_j = \frac{c_j}{N_j} \quad \text{(to the normalized polynomials)}
      \quad \text{and} \quad
      c_j = c'_j N_j \quad \text{(to the not normalized polynomials)}

Rotation
--------

Let the frame of the pupil be rotated by an angle :math:`\phi`: the point of polar angle :math:`\theta` in the new
frame is the point of polar angle :math:`\theta + \phi` in the initial frame, so the function in the new frame is
:math:`f'(\rho, \theta) = f(\rho, \theta + \phi)`. The rotationally symmetric polynomials (:math:`m = 0`) are unchanged,
and for each pair of polynomials :math:`Z_{n}^{m}` and :math:`Z_{n}^{-m}` (:math:`m > 0`) with the coefficients
:math:`a` and :math:`b`:

.. math::

    a \cos(m(\theta + \phi)) + b \sin(m(\theta + \phi))
    = \left(a \cos(m\phi) + b \sin(m\phi)\right) \cos(m\theta) + \left(b \cos(m\phi) - a \sin(m\phi)\right) \sin(m\theta)

so the new coefficients are (:func:`pyzernike.zernike_coefficients_rotate`):

.. math::

    a' = a \cos(m\phi) + b \sin(m\phi)
    \quad \text{and} \quad
    b' = b \cos(m\phi) - a \sin(m\phi)

The normalization factors are the same for :math:`m` and :math:`-m`, so the relation holds for the normalized and the
not normalized coefficients. A rotation is an isometry of the disk: the RMS of :math:`f` is unchanged.

Change of pupil
---------------

Measurements made on pupils of different sizes or positions are compared by expressing them in the same pupil. Let
the new pupil be the disk of center :math:`(x_0, y_0)` and radius :math:`s`, in the normalized coordinates of the
initial pupil (of radius 1), with its axes rotated by :math:`\phi`. The point :math:`p'` of the new unit disk is the
point of the initial pupil:

.. math::

    q(p') = \begin{pmatrix} x_0 \\ y_0 \end{pmatrix} + s \begin{pmatrix} \cos\phi & -\sin\phi \\ \sin\phi & \cos\phi \end{pmatrix} p'

and the function in the new pupil is :math:`f'(p') = f(q(p'))`. The map :math:`p' \mapsto q` is affine and the
Zernike polynomials up to the order :math:`N` span the polynomials of degree :math:`N` in the Cartesian coordinates,
so :math:`f'` is **exactly** a sum of Zernike polynomials up to the same order :math:`N`. Its coefficients are
:math:`c' = T c`, where the matrix :math:`T` is given by the orthogonality relations of :doc:`conventions`:

.. math::

    T_{ij} = N_i^2 \, \frac{1}{\pi} \int_{0}^{2\pi} \int_{0}^{1} Z_i(p') \, Z_j(q(p')) \, \rho' \, d\rho' \, d\theta'

The integrand is a polynomial of degree :math:`2N` in :math:`p'`, so the integral is computed exactly (up to the
rounding errors) with a product quadrature: a Gauss-Legendre rule of :math:`N + 2` nodes in :math:`\rho'` (exact for
the polynomials of degree :math:`2N + 3`, the weight :math:`\rho'` included) and a uniform rule of :math:`2N + 2`
nodes in :math:`\theta'` (exact for the trigonometric polynomials of degree :math:`2N + 1`). The matrix is cached for
the last parameters (:func:`pyzernike.zernike_coefficients_transform`). For the normalized coefficients,
:math:`c'_{norm} = D^{-1} T D \, c_{norm}` with :math:`D = \operatorname{diag}(N_j)`.

For example, the defocus seen on the central part of radius :math:`s = 0.5` of the pupil is
:math:`Z_2^0(0.5\rho) = 0.5\rho^2 - 1 = 0.25 \, Z_2^0(\rho) - 0.75`: a smaller defocus and a piston. On a decentered
sub-aperture, a defocus gives a tilt, and more generally an aberration of order :math:`n` gives aberrations of all the
lower orders.

.. note::

    If the new pupil is not inside the initial pupil (:math:`\sqrt{x_0^2 + y_0^2} + s > 1`), :math:`f` is extrapolated
    outside the initial pupil with its polynomial expression, which is usually not meaningful for measured data.

Statistics
----------

**Root mean square.** The RMS of :math:`f` around its mean on the unit disk follows from the orthogonality relations
(only the piston has a non-zero mean):

.. math::

    \sigma^2 = \frac{1}{\pi} \int_{0}^{2\pi} \int_{0}^{1} \left( f - \bar{f} \right)^2 \rho \, d\rho \, d\theta
    = \sum_{j > 0} \frac{c_j^2}{N_j^2}

which is :math:`\sum_{j > 0} c_j^2` for the normalized coefficients (:func:`pyzernike.zernike_rms`). It is exact,
without sampling. As the mapping of a full ellipse onto the disk is affine, it is also the RMS on the ellipse for the
polynomials of :func:`pyzernike.xy_zernike_polynomial` with :math:`h = 0`.

**Peak-to-valley.** The peak-to-valley :math:`\max f - \min f` on the disk has no closed-form expression. It is
computed on a polar grid (:func:`pyzernike.zernike_peak_to_valley`), which gives a lower bound converging quickly with
the resolution of the grid.

**Strehl ratio.** The Strehl ratio :math:`S` is the ratio of the peak intensity of the point spread function to the
peak intensity of the aberration-free system. For small aberrations, it only depends on the RMS :math:`\sigma` of the
wavefront, with the extended Maréchal approximation [Mahajan1983]_ (:func:`pyzernike.zernike_strehl_ratio`):

.. math::

    S \approx \exp\left( -\left( \frac{2 \pi \sigma}{\lambda} \right)^2 \right)

The approximation is accurate for :math:`S \gtrsim 0.3`. The Maréchal criterion of a diffraction-limited system,
:math:`\sigma \leq \lambda / 14`, corresponds to :math:`S \geq 0.8`.