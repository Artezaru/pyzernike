Computation of the polynomials
==============================

.. contents:: Table of Contents
   :local:
   :depth: 2

Explicit sum
------------

The radial polynomials are computed from their definition (see :doc:`conventions`):

.. math::

    R_{n}^{m}(\rho) = \sum_{k=0}^{(n-m)/2} \frac{(-1)^k (n-k)!}{k! ((n+m)/2 - k)! ((n-m)/2 - k)!} \rho^{n-2k}

The coefficients of the radial polynomial are computed exactly with integer arithmetic (``math.comb``), without forming any factorial,
as a product of two binomial coefficients:

.. math::

    \frac{(n-k)!}{k! ((n+m)/2 - k)! ((n-m)/2 - k)!} = \binom{n-k}{k} \binom{n-2k}{(n-m)/2 - k}

The coefficients are exact and cannot overflow. They are rounded only once, when converted to floating point values,
and the sum is evaluated with the Horner scheme in :math:`\rho^2`.

.. note::

    Even with exact coefficients, the alternating sum suffers from cancellation between its large terms for high orders:
    in double precision, the absolute error stays below :math:`10^{-10}` up to :math:`n \approx 20` for :math:`m = 0`,
    and grows quickly beyond. For high orders, the recurrence relation described below must be preferred
    (``use_recursive=True``).


Derivatives
-----------

The derivative of order :math:`a` of the Zernike polynomial with respect to :math:`\rho` and of order :math:`b`
with respect to :math:`\theta` is defined as follows:

.. math::

    \frac{\partial^{a+b} Z_{n}^{m}}{\partial \rho^{a} \partial \theta^{b}}(\rho, \theta) = \begin{cases}
        \frac{d^a R_{n}^{m}}{d\rho^a}(\rho) \, m^{b} \cos\left(m\theta + b\frac{\pi}{2}\right) & \text{if } m > 0 \\
        \frac{d^a R_{n}^{0}}{d\rho^a}(\rho) \, \delta_{b,0} & \text{if } m = 0 \\
        \frac{d^a R_{n}^{|m|}}{d\rho^a}(\rho) \, |m|^{b} \sin\left(|m|\theta + b\frac{\pi}{2}\right) & \text{if } m < 0
    \end{cases}

The phase shift :math:`b\frac{\pi}{2}` is applied exactly by permuting and negating the cosine and the sine
according to :math:`b \bmod 4`.

The derivative of order :math:`a` of the radial Zernike polynomial is defined as follows:

.. math::

    \frac{d^{a}R_{n}^{m}(\rho)}{d\rho^{a}} = \sum_{k=0}^{(n-m)/2} \frac{(-1)^k (n-k)!}{k! ((n+m)/2 - k)! ((n-m)/2 - k)!} (n-2k) (n-2k-1) \ldots (n-2k-a+1) \rho^{n-2k-a}

The factor :math:`(n-2k) (n-2k-1) \ldots (n-2k-a+1) = \frac{(n-2k)!}{(n-2k-a)!}` is also computed exactly with integer arithmetic (``math.perm``).



Recurrence relation
-------------------

When all the polynomials up to a radial order :math:`N` are needed, they are computed with the three-term recurrence relation of
Shakibaei and Paramesran [Shakibaei2013]_:

.. math::

    R_{n}^{m}(\rho) = \rho \, A_{n}^{m}(\rho) - R_{n-2}^{m}(\rho) \quad \text{with} \quad A_{n}^{m}(\rho) = R_{n-1}^{|m-1|}(\rho) + R_{n-1}^{m+1}(\rho)

for :math:`n \geq 1`, :math:`0 \leq m \leq n` and :math:`(n - m)` even, with the initial condition
:math:`R_{0}^{0}(\rho) = 1` and the convention :math:`R_{n}^{m}(\rho) = 0` whenever :math:`m > n`.
With this convention, the same relation also gives the diagonal terms :math:`R_{n}^{n}(\rho) = \rho^{n}`.

The polynomials are built row by row (one row per radial order :math:`n`), each row only depending on
the rows :math:`n-1` and :math:`n-2`. Each polynomial costs one multiplication and two additions whatever
its order, so that all the polynomials up to the order :math:`N` are computed in :math:`O(N^2)` operations
per point (instead of :math:`O(N^3)` with the explicit sum). The recurrence only combines polynomials
bounded by 1 on :math:`[0, 1]`, so its accuracy stays close to the machine precision at any order.

Differentiating the relation :math:`a` times with the Leibniz rule
:math:`\frac{d^a}{d\rho^a}\left[\rho f(\rho)\right] = \rho f^{(a)}(\rho) + a f^{(a-1)}(\rho)`
gives a recurrence of the same form for the derivatives:

.. math::

    \frac{d^a R_{n}^{m}}{d\rho^a} = \rho \, \frac{d^a A_{n}^{m}}{d\rho^a} + a \, \frac{d^{a-1} A_{n}^{m}}{d\rho^{a-1}} - \frac{d^a R_{n-2}^{m}}{d\rho^a}

with the diagonal terms :math:`\frac{d^a R_{n}^{n}}{d\rho^a} = \frac{n!}{(n-a)!} \rho^{n-a}` and
:math:`\frac{d^a R_{n}^{m}}{d\rho^a} = 0` for :math:`a > n`. The derivative orders :math:`0, 1, \dots, a`
are computed together, since the order :math:`a` requires the order :math:`a - 1`.


Both methods are available in :func:`pyzernike.zernike_polynomial_up_to_order`,
:func:`pyzernike.xy_zernike_polynomial_up_to_order`, :func:`pyzernike.zernike_evaluate` and the fit functions with the
argument ``use_recursive``. The functions computing given polynomials :math:`(n, m)`
(:func:`pyzernike.zernike_polynomial`, :func:`pyzernike.radial_polynomial`) use the explicit sum. The gallery example
*Accuracy and speed at high orders* compares the two methods.