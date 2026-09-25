# Copyright 2025 Artezaru
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from numbers import Integral, Real
from typing import List, Optional, Sequence, Union

import numpy

from .core.parsing import _as_common_floating_arrays, _broadcast_integers
from .core.core_polynomial import _core_polynomials


def zernike_polynomial(
    rho: numpy.ndarray,
    theta: numpy.ndarray,
    n: Union[Integral, Sequence[Integral], numpy.ndarray],
    m: Union[Integral, Sequence[Integral], numpy.ndarray],
    rho_derivative: Optional[Union[Integral, Sequence[Integral], numpy.ndarray]] = 0,
    theta_derivative: Optional[Union[Integral, Sequence[Integral], numpy.ndarray]] = 0,
    default: Real = numpy.nan,
    *,
    precompute: bool = True,
    to_list: bool = True,
) -> Union[numpy.ndarray, List[numpy.ndarray]]:
    r"""
    Compute the Zernike polynomials :math:`Z_{n}^{m}(\rho, \theta)` and their derivatives.

    The Zernike polynomials [1]_ are defined on the unit disk for :math:`|m| \leq n` and :math:`n - m` even by:

    .. math::

        Z_{n}^{m}(\rho, \theta) = \begin{cases}
            R_{n}^{m}(\rho) \cos(m \theta) & \text{if } m \geq 0 \\
            R_{n}^{|m|}(\rho) \sin(|m| \theta) & \text{if } m < 0
        \end{cases}

    where :math:`R_{n}^{m}` is the radial polynomial (see :func:`pyzernike.radial_polynomial`). The
    polynomials are not normalized: :math:`Z_{n}^{m}(1, \theta)` has an amplitude of 1. The normalized
    polynomials of Noll [2]_ are obtained by multiplying by :math:`\sqrt{2(n+1)}` (:math:`m \neq 0`)
    or :math:`\sqrt{n+1}` (:math:`m = 0`).

    The derivative of order :math:`a` with respect to :math:`\rho` and :math:`b` with respect to :math:`\theta` is:

    .. math::

        \frac{\partial^{a+b} Z_{n}^{m}}{\partial \rho^{a} \partial \theta^{b}} = \begin{cases}
            \frac{d^a R_{n}^{m}}{d\rho^a}(\rho) \, m^{b} \cos\left(m\theta + b\frac{\pi}{2}\right) & \text{if } m > 0 \\
            \frac{d^a R_{n}^{0}}{d\rho^a}(\rho) \, \delta_{b,0} & \text{if } m = 0 \\
            \frac{d^a R_{n}^{|m|}}{d\rho^a}(\rho) \, |m|^{b} \sin\left(|m|\theta + b\frac{\pi}{2}\right) & \text{if } m < 0
        \end{cases}

    The coefficients of the radial part are computed exactly with integer arithmetic, and the sum is
    evaluated with the Horner scheme (see :func:`pyzernike.core._core_polynomials`).

    - If :math:`|m| > n` or :math:`n - m` is odd, the polynomial is zero.
    - If :math:`a > n`, the derivative is zero.
    - Where :math:`\rho` is outside :math:`[0, 1]`, or :math:`\rho` or :math:`\theta` is not finite,
      the output is set to ``default``.

    **Shape of the inputs and of the output.** ``n``, ``m``, ``rho_derivative`` and ``theta_derivative``
    can each be a single integer or a sequence of integers (list, tuple or 1D numpy array):

    - If all of them are integers, a single polynomial is computed and returned as an array with the
      shape of ``rho``.
    - If at least one of them is a sequence, the integers are repeated to the length :math:`N` of the
      sequences (which must all have the same length), and :math:`N` polynomials are computed. They are
      returned as a list of :math:`N` arrays with the shape of ``rho`` (``to_list=True``), or as a single
      array of shape :math:`(N, \dots)` (``to_list=False``).

    .. note::

        The explicit sum of the radial part suffers from cancellation between its large alternating terms
        for high orders. In double precision, the absolute error stays below :math:`10^{-10}` up to
        :math:`n \approx 20` for :math:`m = 0`, and grows quickly beyond.

    .. seealso::

        - :func:`pyzernike.radial_polynomial` for the radial polynomials :math:`R_{n}^{m}(\rho)`.
        - :func:`pyzernike.zernike_symbolic` for the symbolic expressions of the Zernike polynomials.
        - :func:`pyzernike.zernike_index_to_order` to convert OSA/ANSI indices to orders :math:`(n, m)`.
        - The page :doc:`../../mathematical_description` for the mathematical description of the Zernike polynomials.

    Parameters
    ----------
    rho : numpy.ndarray
        The radial coordinates :math:`\rho` with any shape ``(...)``.

    theta : numpy.ndarray
        The angular coordinates :math:`\theta` in radians, with the same shape as ``rho``.
        If ``rho`` and ``theta`` are floating point arrays of the same dtype (e.g. ``numpy.float32``),
        this dtype is kept. Otherwise, both are converted to ``numpy.float64``.

    n : Union[Integral, Sequence[Integral], numpy.ndarray]
        The radial order(s) :math:`n`.

    m : Union[Integral, Sequence[Integral], numpy.ndarray]
        The azimuthal frequency(ies) :math:`m`.

    rho_derivative : Optional[Union[Integral, Sequence[Integral], numpy.ndarray]], optional
        The order(s) :math:`a \geq 0` of the derivative with respect to :math:`\rho`. Default is 0 (no derivative).
        None is accepted and means 0.

    theta_derivative : Optional[Union[Integral, Sequence[Integral], numpy.ndarray]], optional
        The order(s) :math:`b \geq 0` of the derivative with respect to :math:`\theta`. Default is 0 (no derivative).
        None is accepted and means 0.

    default : Real, optional
        The value set where :math:`\rho` is outside :math:`[0, 1]`, or :math:`\rho` or :math:`\theta` is not finite.
        Default is ``numpy.nan``.

    precompute : bool, optional
        If True, the powers of :math:`\rho` and the cosine and sine terms are cached and shared between the
        polynomials, which is faster when several polynomials are computed at once. If False, they are
        recomputed for each polynomial, which uses less memory. Default is True.

    to_list : bool, optional
        Only used when several polynomials are computed. If True, the output is a list of arrays.
        If False, the output is a single array of shape :math:`(N, \dots)`. Default is True.

    Returns
    -------
    Union[numpy.ndarray, List[numpy.ndarray]]
        - If ``n``, ``m``, ``rho_derivative`` and ``theta_derivative`` are integers: an array with the shape of ``rho``.
        - Otherwise, if ``to_list`` is True: a list of :math:`N` arrays with the shape of ``rho``.
        - Otherwise: an array of shape :math:`(N, \dots)`, where ``output[i]`` corresponds to
          ``(n[i], m[i], rho_derivative[i], theta_derivative[i])``.

        The dtype is the common floating point dtype of ``rho`` and ``theta`` (``numpy.float64`` by default).

    Raises
    ------
    TypeError
        If ``rho`` or ``theta`` can not be converted to a floating point array, if ``n``, ``m``,
        ``rho_derivative`` or ``theta_derivative`` is not an integer or a sequence of integers, if
        ``default`` is not a real number, or if ``precompute`` or ``to_list`` is not a boolean.

    ValueError
        If ``rho`` and ``theta`` do not have the same shape, if the sequences do not have the same
        length, or if a derivative order is negative.

    References
    ----------
    .. [1] F. Zernike, "Beugungstheorie des Schneidenverfahrens und seiner verbesserten Form, der
       Phasenkontrastmethode," Physica 1(7-12), 689-704 (1934). https://doi.org/10.1016/S0031-8914(34)80259-5

    .. [2] R. J. Noll, "Zernike polynomials and atmospheric turbulence," Journal of the Optical Society
       of America 66(3), 207-211 (1976). https://doi.org/10.1364/JOSA.66.000207

    Examples
    --------
    Compute the defocus :math:`Z_{2}^{0}(\rho, \theta)` on a polar grid:

    .. code-block:: python

        import numpy
        from pyzernike import zernike_polynomial

        rho, theta = numpy.meshgrid(numpy.linspace(0, 1, 100), numpy.linspace(0, 2 * numpy.pi, 100))
        Z20 = zernike_polynomial(rho, theta, n=2, m=0)  # array of shape (100, 100)

    Compute :math:`Z_{2}^{2}` and its gradient in polar coordinates (``n`` and ``m`` are repeated):

    .. code-block:: python

        Z, dZ_drho, dZ_dtheta = zernike_polynomial(rho, theta, n=2, m=2, rho_derivative=[0, 1, 0], theta_derivative=[0, 0, 1])

    Compute the first 15 polynomials in the OSA/ANSI ordering as a single array:

    .. code-block:: python

        from pyzernike import zernike_index_to_order

        n, m = zernike_index_to_order(range(15))
        Z = zernike_polynomial(rho, theta, n=n, m=m, to_list=False)  # array of shape (15, 100, 100)

    """
    rho, theta = _as_common_floating_arrays(rho=rho, theta=theta)
    if not isinstance(default, Real):
        raise TypeError("default must be a real number.")
    if not isinstance(precompute, bool):
        raise TypeError("precompute must be a boolean.")
    if not isinstance(to_list, bool):
        raise TypeError("to_list must be a boolean.")

    if rho_derivative is None:
        rho_derivative = 0
    if theta_derivative is None:
        theta_derivative = 0

    indices, single = _broadcast_integers(n=n, m=m, rho_derivative=rho_derivative, theta_derivative=theta_derivative)
    if any(a < 0 for a in indices["rho_derivative"]):
        raise ValueError("rho_derivative must contain non-negative integers.")
    if any(b < 0 for b in indices["theta_derivative"]):
        raise ValueError("theta_derivative must contain non-negative integers.")

    valid_mask = numpy.isfinite(rho) & numpy.isfinite(theta) & (rho >= 0) & (rho <= 1)

    output = numpy.full((len(indices["n"]),) + rho.shape, default, dtype=rho.dtype)
    output[:, valid_mask] = _core_polynomials(
        rho[valid_mask],
        theta[valid_mask],
        indices["n"],
        indices["m"],
        indices["rho_derivative"],
        indices["theta_derivative"],
        False,
        precompute,
    )

    if single:
        return output[0]
    if to_list:
        return list(output)
    return output