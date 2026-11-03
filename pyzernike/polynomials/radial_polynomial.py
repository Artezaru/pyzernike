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

from numbers import Integral
from typing import List, Optional, Sequence, Union

import numpy

from ..core.parsing import _as_default_value, _as_floating_array, _broadcast_indices, _check_boolean, _format_output
from ..core.core_polynomial import _core_polynomials


def radial_polynomial(
    rho: numpy.ndarray,
    n: Union[Integral, Sequence[Integral], numpy.ndarray],
    m: Union[Integral, Sequence[Integral], numpy.ndarray],
    rho_derivative: Optional[Union[Integral, Sequence[Integral], numpy.ndarray]] = 0,
    *,
    default: Union[int, float] = numpy.nan,
    precompute: bool = True,
    to_list: bool = True,
) -> Union[numpy.ndarray, List[numpy.ndarray]]:
    r"""
    Compute the radial Zernike polynomials :math:`R_{n}^{m}(\rho)` and their derivatives.

    The radial Zernike polynomials [1]_ are defined for :math:`0 \leq m \leq n` and :math:`n - m` even by:

    .. math::

        R_{n}^{m}(\rho) = \sum_{k=0}^{\frac{n-m}{2}} \frac{(-1)^k (n-k)!}{k! \left(\frac{n+m}{2} - k\right)! \left(\frac{n-m}{2} - k\right)!} \rho^{n-2k}

    and the :math:`a`-th derivative with respect to :math:`\rho` is obtained by differentiating each term:

    .. math::

        \frac{d^a R_{n}^{m}}{d\rho^a}(\rho) = \sum_{k} \frac{(-1)^k (n-k)!}{k! \left(\frac{n+m}{2} - k\right)! \left(\frac{n-m}{2} - k\right)!} \frac{(n-2k)!}{(n-2k-a)!} \rho^{n-2k-a}

    where the sum runs over the terms with :math:`n - 2k \geq a`.

    The coefficients are computed exactly with integer arithmetic, and the sum is evaluated with the
    Horner scheme (see the page :doc:`../../mathematical_description/polynomials`). The radial polynomials are never normalized (the
    normalization factors of Noll are defined for the full Zernike polynomials only).

    - If :math:`m < 0`, :math:`m > n` or :math:`n - m` is odd, the polynomial is zero.
    - If :math:`a > n`, the derivative is zero.
    - Where :math:`\rho` is outside :math:`[0, 1]` or is not finite, the output is set to ``default``.

    **Shape of the inputs and of the output.** ``n``, ``m`` and ``rho_derivative`` can each be a single
    integer or a sequence of integers (list, tuple or 1D numpy array):

    - If all of them are integers, a single polynomial is computed and returned as an array with the
      shape of ``rho``.
    - If at least one of them is a sequence, the integers are repeated to the length :math:`N` of the
      sequences (which must all have the same length), and :math:`N` polynomials are computed. They are
      returned as a list of :math:`N` arrays with the shape of ``rho`` (``to_list=True``), or as a single
      array of shape :math:`(N, \dots)` (``to_list=False``).

    .. note::

        The explicit sum suffers from cancellation between its large alternating terms for high orders.
        In double precision, the absolute error stays below :math:`10^{-10}` up to :math:`n \approx 20`
        for :math:`m = 0`, and grows quickly beyond. For high orders, the radial polynomials can be obtained
        with the recurrence relation as :math:`R_{n}^{m}(\rho) = Z_{n}^{m}(\rho, 0)` (:math:`m \geq 0`), with
        :func:`pyzernike.zernike_polynomial_up_to_order` and ``use_recursive=True``.

    .. seealso::

        - :func:`pyzernike.zernike_polynomial` for the full Zernike polynomials :math:`Z_{n}^{m}(\rho, \theta)`.
        - :func:`pyzernike.radial_symbolic` for the symbolic expressions of the radial polynomials.
        - The page :doc:`../../mathematical_description/conventions` for the definition of the Zernike polynomials.
        - The page :doc:`../../mathematical_description/polynomials` for the explicit sum, the derivatives and the recurrence relation.

    Parameters
    ----------
    rho : numpy.ndarray
        The radial coordinates :math:`\rho` with any shape ``(...)``. Floating point arrays keep their dtype
        (e.g. ``numpy.float32``), any other input is converted to ``numpy.float64``.

    n : Union[Integral, Sequence[Integral], numpy.ndarray]
        The radial order(s) :math:`n`.

    m : Union[Integral, Sequence[Integral], numpy.ndarray]
        The azimuthal frequency(ies) :math:`m`.

    rho_derivative : Optional[Union[Integral, Sequence[Integral], numpy.ndarray]], optional
        The order(s) :math:`a \geq 0` of the derivative with respect to :math:`\rho`. Default is 0 (no derivative).
        None is accepted and means 0.

    default : Union[int, float], optional
        The value set where :math:`\rho` is outside :math:`[0, 1]` or is not finite. Integers and floating point
        numbers (Python or numpy, including ``numpy.nan`` and ``numpy.inf``) are accepted and converted to float.
        Booleans are not accepted. Default is ``numpy.nan``.

    precompute : bool, optional
        If True, the powers of :math:`\rho` are cached and shared between the polynomials, which is faster
        when several polynomials are computed at once. If False, they are recomputed for each polynomial,
        which uses less memory. Default is True.

    to_list : bool, optional
        Only used when several polynomials are computed. If True, the output is a list of arrays.
        If False, the output is a single array of shape :math:`(N, \dots)`. Default is True.

    Returns
    -------
    Union[numpy.ndarray, List[numpy.ndarray]]
        - If ``n``, ``m`` and ``rho_derivative`` are integers: an array with the shape of ``rho``.
        - Otherwise, if ``to_list`` is True: a list of :math:`N` arrays with the shape of ``rho``.
        - Otherwise: an array of shape :math:`(N, \dots)`, where ``output[i]`` corresponds to
          ``(n[i], m[i], rho_derivative[i])``.

        The dtype is the floating point dtype of ``rho`` (``numpy.float64`` by default).

    Raises
    ------
    TypeError
        If ``rho`` can not be converted to a floating point array, if ``n``, ``m`` or ``rho_derivative``
        is not an integer or a sequence of integers, if ``default`` is not an integer or a floating point number, or if
        ``precompute`` or ``to_list`` is not a boolean.

    ValueError
        If the sequences ``n``, ``m`` and ``rho_derivative`` do not have the same length, or if a
        derivative order is negative.

    References
    ----------
    .. [1] F. Zernike, "Beugungstheorie des Schneidenverfahrens und seiner verbesserten Form, der
       Phasenkontrastmethode," Physica 1(7-12), 689-704 (1934). https://doi.org/10.1016/S0031-8914(34)80259-5

    Examples
    --------
    Compute :math:`R_{2}^{0}(\rho)`:

    .. code-block:: python

        import numpy
        from pyzernike import radial_polynomial

        rho = numpy.linspace(0, 1, 100)
        R20 = radial_polynomial(rho, n=2, m=0)  # array of shape (100,)

    Compute :math:`R_{2}^{0}(\rho)` and its first two derivatives (``n`` and ``m`` are repeated):

    .. code-block:: python

        R20, dR20, d2R20 = radial_polynomial(rho, n=2, m=0, rho_derivative=[0, 1, 2])

    Compute all the radial polynomials of order 4 as a single array:

    .. code-block:: python

        R4 = radial_polynomial(rho, n=4, m=[0, 2, 4], to_list=False)  # array of shape (3, 100)

    """
    rho = _as_floating_array(rho, "rho")
    default = _as_default_value(default)
    precompute = _check_boolean(precompute, "precompute")
    to_list = _check_boolean(to_list, "to_list")
    indices, single = _broadcast_indices(
        integers={"n": n, "m": m},
        orders={"rho_derivative": rho_derivative},
    )

    valid_mask = numpy.isfinite(rho) & (rho >= 0) & (rho <= 1)

    output = numpy.full((len(indices["n"]),) + rho.shape, default, dtype=rho.dtype)
    output[:, valid_mask] = _core_polynomials(
        rho=rho[valid_mask],
        theta=None,
        n=indices["n"],
        m=indices["m"],
        rho_derivative=indices["rho_derivative"],
        theta_derivative=None,
        flag_radial=True,
        precompute=precompute,
        normalize=False,
    )

    return _format_output(output, single, to_list)