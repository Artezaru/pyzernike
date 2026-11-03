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

from ..core.parsing import (
    _as_common_floating_arrays,
    _as_default_value,
    _as_osa_coefficients,
    _broadcast_indices,
    _check_boolean,
    _format_output,
)
from ..core.core_evaluate import _core_evaluate


def zernike_evaluate(
    coefficients: Union[Sequence[float], numpy.ndarray],
    rho: numpy.ndarray,
    theta: numpy.ndarray,
    rho_derivative: Optional[Union[Integral, Sequence[Integral], numpy.ndarray]] = 0,
    theta_derivative: Optional[Union[Integral, Sequence[Integral], numpy.ndarray]] = 0,
    *,
    default: Union[int, float] = numpy.nan,
    precompute: bool = True,
    use_recursive: bool = False,
    normalize: bool = False,
    to_list: bool = True,
) -> Union[numpy.ndarray, List[numpy.ndarray]]:
    r"""
    Evaluate a sum of Zernike polynomials :math:`f = \sum_j c_j Z_j` and its derivatives in polar coordinates.

    .. math::

        \frac{\partial^{a+b} f}{\partial \rho^{a} \partial \theta^{b}}(\rho, \theta)
        = \sum_{j=0}^{P-1} c_j \frac{\partial^{a+b} Z_j}{\partial \rho^{a} \partial \theta^{b}}(\rho, \theta)

    where :math:`Z_j` is the Zernike polynomial of OSA/ANSI index :math:`j` (see
    :func:`pyzernike.zernike_index_to_order`). The coefficients must contain all the polynomials up to a maximum
    radial order :math:`N`, so their number must be :math:`P = \frac{(N+1)(N+2)}{2}` (1, 3, 6, 10, 15, ...): pad
    them with zeros if needed. Coefficients in the Noll or Fringe indexing can be converted with
    :func:`pyzernike.zernike_coefficients_convert`.

    The polynomials are computed by chunks of points, so that the :math:`P` arrays of the polynomials are never
    stored for all the points at once (unlike :func:`pyzernike.zernike_polynomial_up_to_order` followed by a sum).

    Where :math:`\rho` is outside :math:`[0, 1]`, or :math:`\rho` or :math:`\theta` is not finite, the output is set
    to ``default``.

    **Shape of the inputs and of the output.** ``rho_derivative`` and ``theta_derivative`` can each be a single
    integer or a sequence of integers (list, tuple or 1D numpy array):

    - If both are integers, the output is an array with the shape of ``rho``.
    - If at least one of them is a sequence, the integer is repeated to the length :math:`D` of the sequence (the
      sequences must have the same length), and the output is a list of :math:`D` arrays with the shape of ``rho``
      (``to_list=True``), or a single array of shape :math:`(D, \dots)` (``to_list=False``).

    .. seealso::

        - :func:`pyzernike.xy_zernike_evaluate` to evaluate a sum of polynomials on an elliptic annulus.
        - :func:`pyzernike.zernike_fit` to compute the coefficients from sampled values.
        - :func:`pyzernike.zernike_coefficients_convert` to convert coefficients between conventions.
        - The page :doc:`../../mathematical_description/coefficients` for the operations on the coefficients.

    Parameters
    ----------
    coefficients : Union[Sequence[float], numpy.ndarray]
        The coefficients :math:`c_j` in the OSA/ANSI ordering, as a 1D array of :math:`P = \frac{(N+1)(N+2)}{2}`
        finite real numbers.

    rho : numpy.ndarray
        The radial coordinates :math:`\rho` with any shape ``(...)``.

    theta : numpy.ndarray
        The angular coordinates :math:`\theta` in radians, with the same shape as ``rho``.
        If ``rho`` and ``theta`` are floating point arrays of the same dtype (e.g. ``numpy.float32``),
        this dtype is kept. Otherwise, both are converted to ``numpy.float64``.

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
        Only used if ``use_recursive`` is False. If True, the powers of :math:`\rho` and the cosine and sine terms
        are cached and shared between the polynomials. Default is True.

    use_recursive : bool, optional
        If True, the polynomials are computed with the recurrence relation (recommended for high orders, see
        :func:`pyzernike.zernike_polynomial_up_to_order`). Default is False.

    normalize : bool, optional
        If True, the coefficients are those of the polynomials normalized by the factors of Noll
        :math:`N_{n}^{m} = \sqrt{(2 - \delta_{m0})(n + 1)}` (see :func:`pyzernike.zernike_normalization`).
        Default is False.

    to_list : bool, optional
        Only used when several derivatives are computed. If True, the output is a list of arrays. If False, the
        output is a single array of shape :math:`(D, \dots)`. Default is True.

    Returns
    -------
    Union[numpy.ndarray, List[numpy.ndarray]]
        - If ``rho_derivative`` and ``theta_derivative`` are integers: an array with the shape of ``rho``.
        - Otherwise, if ``to_list`` is True: a list of :math:`D` arrays with the shape of ``rho``.
        - Otherwise: an array of shape :math:`(D, \dots)`, where ``output[d]`` is the derivative
          ``(rho_derivative[d], theta_derivative[d])``.

        The dtype is the common floating point dtype of ``rho`` and ``theta`` (``numpy.float64`` by default).

    Raises
    ------
    TypeError
        If ``coefficients``, ``rho`` or ``theta`` can not be converted to a floating point array, if
        ``rho_derivative`` or ``theta_derivative`` is not an integer or a sequence of integers, if ``default`` is
        not an integer or a floating point number, or if ``precompute``, ``use_recursive``, ``normalize`` or
        ``to_list`` is not a boolean.

    ValueError
        If ``coefficients`` is not a 1D array of finite values whose length is :math:`\frac{(N+1)(N+2)}{2}`, if
        ``rho`` and ``theta`` do not have the same shape, if the sequences do not have the same length, or if a
        derivative order is negative.

    Examples
    --------
    Evaluate a sum of defocus and coma on a polar grid:

    .. code-block:: python

        import numpy
        from pyzernike import zernike_evaluate, zernike_order_to_index

        coefficients = numpy.zeros(10)  # all the polynomials up to the order 3
        coefficients[zernike_order_to_index(n=[2, 3], m=[0, 1])] = [0.5, -0.2]

        rho, theta = numpy.meshgrid(numpy.linspace(0, 1, 100), numpy.linspace(0, 2 * numpy.pi, 100))
        W = zernike_evaluate(coefficients, rho, theta)  # array of shape (100, 100)

    Compute its gradient in polar coordinates:

    .. code-block:: python

        dW_drho, dW_dtheta = zernike_evaluate(coefficients, rho, theta, rho_derivative=[1, 0], theta_derivative=[0, 1])

    """
    coefficients, order = _as_osa_coefficients(coefficients)
    rho, theta = _as_common_floating_arrays(rho=rho, theta=theta)
    default = _as_default_value(default)
    precompute = _check_boolean(precompute, "precompute")
    use_recursive = _check_boolean(use_recursive, "use_recursive")
    normalize = _check_boolean(normalize, "normalize")
    to_list = _check_boolean(to_list, "to_list")
    derivatives, single = _broadcast_indices(
        integers={},
        orders={"rho_derivative": rho_derivative, "theta_derivative": theta_derivative},
    )

    valid_mask = numpy.isfinite(rho) & numpy.isfinite(theta) & (rho >= 0) & (rho <= 1)

    output = numpy.full((len(derivatives["rho_derivative"]),) + rho.shape, default, dtype=rho.dtype)
    output[:, valid_mask] = _core_evaluate(
        rho=rho[valid_mask],
        theta=theta[valid_mask],
        coefficients=coefficients,
        order=order,
        rho_derivative=derivatives["rho_derivative"],
        theta_derivative=derivatives["theta_derivative"],
        use_recursive=use_recursive,
        normalize=normalize,
        precompute=precompute,
    )

    return _format_output(output, single, to_list)