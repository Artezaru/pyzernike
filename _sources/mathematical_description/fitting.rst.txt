Fitting
=======

.. contents:: Table of Contents
   :local:
   :depth: 2

Linear least squares
--------------------

A function :math:`f` sampled at the points :math:`(\rho_k, \theta_k)`, with the values :math:`f_k` and the weights
:math:`w_k > 0`, is approximated by the polynomials up to the order :math:`N` (:func:`pyzernike.zernike_fit`):

.. math::

    \min_{c} \sum_{k} w_k \left( f_k - \sum_{j=0}^{P-1} c_j Z_j(\rho_k, \theta_k) \right)^2

The problem is linear in the coefficients :math:`c`. With the design matrix :math:`A_{kj} = \sqrt{w_k} \, Z_j(\rho_k, \theta_k)`
and :math:`b_k = \sqrt{w_k} \, f_k`, the solution minimizes :math:`\|A c - b\|^2`. It is computed with an orthogonal
decomposition of :math:`A` (``numpy.linalg.lstsq``), and not with the normal equations :math:`A^\top A c = A^\top b`,
whose condition number is the square of the condition number of :math:`A`.

If the samples do not determine all the coefficients (for example if they do not cover the disk, or if there are
fewer samples than polynomials), :math:`A` is rank deficient: a warning is emitted and the solution of minimum norm is
returned. The samples outside the disk, with non-finite values or with a zero weight are ignored.

On an elliptic annulus of known parameters (:func:`pyzernike.xy_zernike_fit`), the points are first converted to their
equivalent polar coordinates :math:`(\rho_{eq}, \theta_{eq})` (see :doc:`elliptic_annulus`), and the same problem is
solved.

Orthogonality and sampling
--------------------------

The least squares solution does not require the polynomials to be orthogonal on the samples: the fitted function is
the same with any basis of the same polynomials (normalized or not). The orthogonality on the continuous disk has
other consequences:

- **Stability of the coefficients.** If the samples are dense and uniform in area, the polynomials are almost
  orthogonal on the samples, :math:`A^\top A` is almost diagonal, and the coefficients almost do not depend on the
  maximum order :math:`N` of the fit: adding higher orders does not change the lower ones.
- **Variance.** With the normalized polynomials, the variance of the fitted function on the disk is
  :math:`\sum_{j > 0} c_j^2` (see :doc:`coefficients`).

For a full ellipse (:math:`h = 0`), the mapping onto the disk is affine, with a constant Jacobian, so the polynomials
stay orthogonal for the area of the ellipse and these properties hold for samples uniform on the ellipse.

For an elliptic annulus (:math:`h > 0`), the mapping is not affine. With the change of variables
:math:`r = h + (1 - h)\rho_{eq}`, the area element of the annulus is, up to a constant factor:

.. math::

    dx \, dy \propto \left(h + (1 - h)\rho_{eq}\right) d\rho_{eq} \, d\theta_{eq}

instead of :math:`\rho_{eq} \, d\rho_{eq} \, d\theta_{eq}`. The polynomials :math:`Zxy_{n}^{m}` are therefore
orthogonal for a weight that is not uniform on the annulus [Navarro2014]_, and the relation between the coefficients
and the variance is only approximate. As the area element does not depend on :math:`\theta_{eq}`, the polynomials of
different azimuthal frequencies :math:`m` stay orthogonal: an orthonormal basis for the area of the annulus would only
mix the polynomials of the same :math:`|m|` (with the radial weight :math:`h + (1 - h)\rho_{eq}` on :math:`[0, 1]`).

.. note::

    On a real pupil sampled on a grid of pixels (with a mask, missing pixels, ...), the polynomials are never exactly
    orthogonal on the samples. The least squares fit gives the right solution in all cases; only the interpretation
    of the individual coefficients requires a dense and uniform sampling.

Non-linear least squares: parameters of the annulus
---------------------------------------------------

When the position, the size or the orientation of the pupil are not known exactly, the parameters :math:`p` of the
annulus become unknowns of a non-linear problem. :func:`pyzernike.xy_zernike_least_squares` builds the residual and
the Jacobian functions of this problem, to be given to any solver (Gauss-Newton, Levenberg-Marquardt,
``scipy.optimize.least_squares``, ...):

.. math::

    r_k(q) = \sqrt{w_k} \left( \sum_j c_j \, Zxy_j(x_k, y_k; p) - f_k \right)
    \quad \text{and} \quad
    J_{ki}(q) = \frac{\partial r_k}{\partial q_i}

where the unknowns :math:`q` are the free parameters :math:`p` of the annulus (and optionally the coefficients
:math:`c`). The derivatives are exact:

.. math::

    \frac{\partial W}{\partial c_j} = Zxy_j
    \quad \text{and} \quad
    \frac{\partial W}{\partial p_i} = \sum_j c_j \, \frac{\partial Zxy_j}{\partial p_i}

with the derivatives with respect to the parameters of :doc:`elliptic_annulus`.

**Points outside the annulus.** The set of the samples inside the annulus changes with :math:`p`, which is not
differentiable, and the cost could decrease only by shrinking the annulus. By default, the polynomials are therefore
evaluated with their polynomial expression at all the samples, also slightly outside the current annulus: the
residuals are smooth functions of :math:`q`, the number of residuals is constant, and the Gauss-Newton algorithm
converges quadratically near the solution.

**Identifiability.** A shift, a scaling or a rotation of the pupil is an affine map, and transforms a polynomial of
degree :math:`N` into another polynomial of degree :math:`N` (see the change of pupil in :doc:`coefficients`). For a
full ellipse (:math:`h = 0`), a change of the parameters :math:`(x_0, y_0, R_x, R_y, \alpha)` is therefore exactly
compensated by a change of the coefficients:

.. math::

    \sum_j c_j \, Zxy_j(x, y; p) = \sum_j c'_j \, Zxy_j(x, y; p') \quad \text{with} \quad c' = T(p, p') \, c

The columns of the Jacobian of these parameters are then linear combinations of the columns of the coefficients: the
Jacobian is singular, and the parameters can not be estimated jointly with the coefficients from the values of
:math:`f` alone. For :math:`h > 0`, the mapping is not affine and the problem is only poorly conditioned. The
parameters of the annulus can be estimated:

- from the values of :math:`f`, if the coefficients are known (e.g. the known aberrations of a reference optics);
- otherwise, from the boundary of the pupil (e.g. on an intensity image), which is not described by the polynomials.