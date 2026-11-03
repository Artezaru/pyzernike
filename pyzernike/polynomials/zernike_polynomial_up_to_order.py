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

from ..core.parsing import _as_common_floating_arrays, _as_default_value, _as_order, _broadcast_indices, _check_boolean
from ..core.core_polynomial import _core_polynomials
from ..core.core_recursive import _core_recursive_up_to_order
from ..core.core_indexing import _core_index_to_order, _core_polynomials_count


def zernike_polynomial_up_to_order(
    rho: numpy.ndarray,
    theta: numpy.ndarray,
    order: Integral,
    rho_derivative: Optional[Union[Integral, Sequence[Integral], numpy.ndarray]] = 0,
    theta_derivative: Optional[Union[Integral, Sequence[Integral], numpy.ndarray]] = 0,
    *,
    default: Union[int, float] = numpy.nan,
    precompute: bool = True,
    use_recursive: bool = False,
    normalize: bool = False,
    to_list: bool = True,
) -> Union[numpy.ndarray, List[List[numpy.ndarray]]]:
    r"""
    Compute all the Zernike polynomials :math:`Z_{n}^{m}(\rho, \theta)` up to a radial order :math:`N`, and their derivatives.

    The Zernike polynomials [1]_ are defined on the unit disk for :math:`|m| \leq n` and :math:`n - m` even by:

    .. math::

        Z_{n}^{m}(\rho, \theta) = \begin{cases}
            R_{n}^{m}(\rho) \cos(m \theta) & \text{if } m \geq 0 \\
            R_{n}^{|m|}(\rho) \sin(|m| \theta) & \text{if } m < 0
        \end{cases}

    and their derivatives are given in :func:`pyzernike.zernike_polynomial`. All the
    :math:`P = \frac{(N+1)(N+2)}{2}` polynomials with :math:`0 \leq n \leq N` are computed, in the
    OSA/ANSI ordering: the polynomial :math:`Z_{n}^{m}` is at the index :math:`j = \frac{n(n+2) + m}{2}`
    (see :func:`pyzernike.zernike_index_to_order`).

    Two methods are available to compute the radial polynomials:

    - ``use_recursive=False``: the explicit sum with exact integer coefficients, evaluated with the Horner
      scheme. The cost is :math:`O(N^3)` operations per
      point, and the cancellation between the large alternating terms of the sum limits the accuracy for
      high orders (the absolute error in double precision grows quickly beyond :math:`n \approx 20`).
    - ``use_recursive=True``: the three-term recurrence relation of Shakibaei and Paramesran [2]_. The cost
      is :math:`O(N^2)` operations per
      point, and the accuracy stays close to the machine precision at any order. This method is
      recommended for high orders.

    Both methods are described in the page :doc:`../../mathematical_description/polynomials`.

    Where :math:`\rho` is outside :math:`[0, 1]`, or :math:`\rho` or :math:`\theta` is not finite, the
    output is set to ``default``.

    **Shape of the inputs and of the output.** ``rho_derivative`` and ``theta_derivative`` can each be a
    single integer or a sequence of integers (list, tuple or 1D numpy array). The integer is repeated to
    the length :math:`D` of the sequence (the sequences must have the same length), and each pair
    ``(rho_derivative[d], theta_derivative[d])`` gives one set of :math:`P` polynomials. The output always
    has a first dimension of length :math:`D` (:math:`D = 1` if both are integers):

    - ``to_list=True``: a list of :math:`D` lists of :math:`P` arrays with the shape of ``rho``.
    - ``to_list=False``: a single array of shape :math:`(D, P, \dots)`.

    .. seealso::

        - :func:`pyzernike.zernike_polynomial` to compute given polynomials :math:`(n, m)`.
        - :func:`pyzernike.zernike_index_to_order` to convert the OSA/ANSI indices to orders :math:`(n, m)`.
        - :func:`pyzernike.zernike_polynomials_count` for the number :math:`P` of polynomials up to an order.
        - :func:`pyzernike.zernike_evaluate` to evaluate a sum of polynomials from its coefficients, without storing
          all the polynomials.
        - The page :doc:`../../mathematical_description/polynomials` for the explicit sum and the recurrence relation.

    Parameters
    ----------
    rho : numpy.ndarray
        The radial coordinates :math:`\rho` with any shape ``(...)``.

    theta : numpy.ndarray
        The angular coordinates :math:`\theta` in radians, with the same shape as ``rho``.
        If ``rho`` and ``theta`` are floating point arrays of the same dtype (e.g. ``numpy.float32``),
        this dtype is kept. Otherwise, both are converted to ``numpy.float64``.

    order : Integral
        The maximum radial order :math:`N \geq 0`.

    rho_derivative : Optional[Union[Integral, Sequence[Integral], numpy.ndarray]], optional
        The order(s) :math:`a \geq 0` of the derivative with respect to :math:`\rho`. Default is 0 (no derivative).
        None is accepted and means 0.

    theta_derivative : Optional[Union[Integral, Sequence[Integral], numpy.ndarray]], optional
        The order(s) :math:`b \geq 0` of the derivative with respect to :math:`\theta`. Default is 0 (no derivative).
        None is accepted and means 0.

    default : Union[int, float], optional
        The value set where :math:`\rho` is outside :math:`[0, 1]`, or :math:`\rho` or :math:`\theta` is not finite.
        Integers and floating point numbers (Python or numpy, including ``numpy.nan`` and ``numpy.inf``) are
        accepted and converted to float. Booleans are not accepted. Default is ``numpy.nan``.

    precompute : bool, optional
        Only used if ``use_recursive`` is False. If True, the powers of :math:`\rho` and the cosine and sine
        terms are cached and shared between the polynomials. If False, they are recomputed for each
        polynomial, which uses less memory. Default is True.

    use_recursive : bool, optional
        If True, the radial polynomials are computed with the recurrence relation instead of the explicit
        sum (faster and more accurate for high orders). Default is False.

    normalize : bool, optional
        If True, the polynomials and their derivatives are multiplied by the normalization factors of Noll
        :math:`N_{n}^{m} = \sqrt{(2 - \delta_{m0})(n + 1)}`, so that they are orthonormal on the unit disk
        (see :func:`pyzernike.zernike_normalization`). The value ``default`` is not normalized. Default is False.

    to_list : bool, optional
        If True, the output is a list of lists of arrays. If False, the output is a single array of shape
        :math:`(D, P, \dots)`. Default is True.

    Returns
    -------
    Union[List[List[numpy.ndarray]], numpy.ndarray]
        The polynomials, where ``output[d][j]`` (``to_list=True``) or ``output[d, j]`` (``to_list=False``) is the
        polynomial of OSA/ANSI index ``j`` for the pair ``(rho_derivative[d], theta_derivative[d])``, with the
        shape of ``rho``. The dtype is the common floating point dtype of ``rho`` and ``theta``
        (``numpy.float64`` by default).

    Raises
    ------
    TypeError
        If ``rho`` or ``theta`` can not be converted to a floating point array, if ``order`` is not an
        integer, if ``rho_derivative`` or ``theta_derivative`` is not an integer or a sequence of integers,
        if ``default`` is not an integer or a floating point number, or if ``precompute``, ``use_recursive``,
        ``normalize`` or ``to_list`` is not a boolean.

    ValueError
        If ``rho`` and ``theta`` do not have the same shape, if ``order`` is negative, if the sequences do
        not have the same length, or if a derivative order is negative.

    References
    ----------
    .. [1] F. Zernike, "Beugungstheorie des Schneidenverfahrens und seiner verbesserten Form, der
       Phasenkontrastmethode," Physica 1(7-12), 689-704 (1934). https://doi.org/10.1016/S0031-8914(34)80259-5

    .. [2] B. H. Shakibaei and R. Paramesran, "Recursive formula to compute Zernike radial polynomials,"
       Optics Letters 38(14), 2487-2489 (2013). https://doi.org/10.1364/OL.38.002487

    Examples
    --------
    Compute all the Zernike polynomials up to the order 3 on a polar grid:

    .. code-block:: python

        import numpy
        from pyzernike import zernike_polynomial_up_to_order, zernike_index_to_order

        rho, theta = numpy.meshgrid(numpy.linspace(0, 1, 100), numpy.linspace(0, 2 * numpy.pi, 100))
        polynomials = zernike_polynomial_up_to_order(rho, theta, order=3)[0]  # list of 10 arrays of shape (100, 100)

        n, m = zernike_index_to_order(range(len(polynomials)))
        for n_j, m_j, Z in zip(n, m, polynomials):
            print(f"Z_{n_j}^{m_j}: max = {Z.max():.3f}")

    Compute all the polynomials up to the order 40 and their gradient in polar coordinates, as a single
    array, with the recurrence relation:

    .. code-block:: python

        Z = zernike_polynomial_up_to_order(rho, theta, order=40, rho_derivative=[0, 1, 0], theta_derivative=[0, 0, 1], use_recursive=True, to_list=False)
        # Z.shape == (3, 861, 100, 100)

    """
    rho, theta = _as_common_floating_arrays(rho=rho, theta=theta)
    order = _as_order(order)
    default = _as_default_value(default)
    precompute = _check_boolean(precompute, "precompute")
    use_recursive = _check_boolean(use_recursive, "use_recursive")
    normalize = _check_boolean(normalize, "normalize")
    to_list = _check_boolean(to_list, "to_list")
    derivatives, _ = _broadcast_indices(
        integers={},
        orders={"rho_derivative": rho_derivative, "theta_derivative": theta_derivative},
    )
    rho_derivative = derivatives["rho_derivative"]
    theta_derivative = derivatives["theta_derivative"]

    n_derivatives = len(rho_derivative)
    n_polynomials = _core_polynomials_count(order)

    valid_mask = numpy.isfinite(rho) & numpy.isfinite(theta) & (rho >= 0) & (rho <= 1)
    all_valid = bool(valid_mask.all())
    rho_valid = rho.ravel() if all_valid else rho[valid_mask]
    theta_valid = theta.ravel() if all_valid else theta[valid_mask]

    if use_recursive:
        values = _core_recursive_up_to_order(
            rho=rho_valid,
            theta=theta_valid,
            order=order,
            rho_derivative=rho_derivative,
            theta_derivative=theta_derivative,
            normalize=normalize,
        )
    else:
        n, m = _core_index_to_order(list(range(n_polynomials)), indexing="OSA")
        values = _core_polynomials(
            rho=rho_valid,
            theta=theta_valid,
            n=n * n_derivatives,
            m=m * n_derivatives,
            rho_derivative=[a for a in rho_derivative for _ in range(n_polynomials)],
            theta_derivative=[b for b in theta_derivative for _ in range(n_polynomials)],
            flag_radial=False,
            precompute=precompute,
            normalize=normalize,
        ).reshape((n_derivatives, n_polynomials, rho_valid.size))

    if all_valid:
        output = values.reshape((n_derivatives, n_polynomials) + rho.shape)
    else:
        output = numpy.full((n_derivatives, n_polynomials) + rho.shape, default, dtype=rho.dtype)
        output[:, :, valid_mask] = values

    if to_list:
        return [list(output[d]) for d in range(n_derivatives)]
    return output