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

import warnings
from numbers import Integral
from typing import Optional, Tuple, Union

import numpy

from ..core.parsing import _as_common_floating_arrays, _as_fit_weights, _as_order, _check_boolean
from ..core.core_fit import _core_fit
from ..core.core_indexing import _core_polynomials_count


def zernike_fit(
    rho: numpy.ndarray,
    theta: numpy.ndarray,
    values: numpy.ndarray,
    order: Integral,
    *,
    weights: Optional[numpy.ndarray] = None,
    use_recursive: bool = False,
    normalize: bool = False,
    return_residuals: bool = False,
) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
    r"""
    Fit a sampled function with the Zernike polynomials up to a radial order :math:`N` by linear least squares.

    The function :math:`f` sampled at the points :math:`(\rho_k, \theta_k)` is approximated by:

    .. math::

        f(\rho, \theta) \approx \sum_{j=0}^{P-1} c_j Z_j(\rho, \theta)

    where :math:`Z_j` is the Zernike polynomial of OSA/ANSI index :math:`j` (see
    :func:`pyzernike.zernike_index_to_order`) and :math:`P = \frac{(N+1)(N+2)}{2}`. The coefficients
    :math:`c_j` minimize the weighted sum of the squared residuals:

    .. math::

        \min_{c} \sum_{k} w_k \left( f_k - \sum_{j} c_j Z_j(\rho_k, \theta_k) \right)^2

    The problem is linear in :math:`c`, so it is solved directly (no iteration) with an orthogonal
    decomposition of the design matrix :math:`A_{kj} = \sqrt{w_k} Z_j(\rho_k, \theta_k)`
    (:func:`numpy.linalg.lstsq`), which is more stable than the normal equations
    :math:`A^\top A c = A^\top f`.

    The points where :math:`\rho` is outside :math:`[0, 1]`, or where :math:`\rho`, :math:`\theta`,
    ``values`` or ``weights`` is not finite, or where the weight is zero, are ignored.

    .. note::

        The least squares solution does not require the polynomials to be orthogonal on the sampled
        points: the reconstructed function is the same with or without normalization. The orthogonality
        on the unit disk only ensures that the coefficients are almost independent of the order :math:`N`
        when the sampling is dense and uniform in area. With ``normalize=True``, the coefficients are
        expressed in the orthonormal basis of Noll, and the variance of :math:`f` on the disk is then
        :math:`\sum_{j > 0} c_j^2` (see :func:`pyzernike.zernike_rms`).

    .. seealso::

        - :func:`pyzernike.xy_zernike_fit` to fit a function sampled in Cartesian coordinates on an elliptic annulus.
        - :func:`pyzernike.zernike_evaluate` to evaluate the fitted function and its derivatives from the coefficients.
        - :func:`pyzernike.zernike_index_to_order` to convert the OSA/ANSI indices to orders :math:`(n, m)`.
        - :func:`pyzernike.zernike_normalization` for the normalization factors of Noll.
        - The page :doc:`../../mathematical_description/fitting` for the least squares fit and the role of the orthogonality.

    Parameters
    ----------
    rho : numpy.ndarray
        The radial coordinates :math:`\rho` of the samples, with any shape ``(...)``.

    theta : numpy.ndarray
        The angular coordinates :math:`\theta` of the samples in radians, with the same shape as ``rho``.

    values : numpy.ndarray
        The sampled values :math:`f_k`, with the same shape as ``rho``.

    order : Integral
        The maximum radial order :math:`N \geq 0` of the fit.

    weights : Optional[numpy.ndarray], optional
        The weights :math:`w_k \geq 0` of the samples, with the same shape as ``rho``. Default is None
        (all the weights are equal to 1).

    use_recursive : bool, optional
        If True, the polynomials are computed with the recurrence relation (recommended for high orders,
        see :func:`pyzernike.zernike_polynomial_up_to_order`). Default is False.

    normalize : bool, optional
        If True, the fit uses the polynomials normalized by the factors of Noll
        :math:`N_{n}^{m} = \sqrt{(2 - \delta_{m0})(n + 1)}`, and the coefficients are those of the
        orthonormal basis. Default is False.

    return_residuals : bool, optional
        If True, the residuals :math:`f_k - \sum_{j} c_j Z_j(\rho_k, \theta_k)` are also returned.
        Default is False.

    Returns
    -------
    coefficients : numpy.ndarray
        The coefficients :math:`c_j` with shape :math:`(P,)`, in the OSA/ANSI ordering, as ``numpy.float64``.

    residuals : numpy.ndarray
        Only if ``return_residuals`` is True. The residuals with the shape of ``rho``, as ``numpy.float64``.
        They are set to ``numpy.nan`` at the ignored points.

    Raises
    ------
    TypeError
        If ``rho``, ``theta``, ``values`` or ``weights`` can not be converted to a floating point array,
        if ``order`` is not an integer, or if ``use_recursive``, ``normalize`` or ``return_residuals`` is
        not a boolean.

    ValueError
        If the arrays do not have the same shape, if ``order`` is negative, if a weight is negative, or if
        the number of valid samples is lower than the number :math:`P` of polynomials.

    Warns
    -----
    RuntimeWarning
        If the design matrix is rank deficient (the samples do not determine all the coefficients, for
        example if they do not cover the disk). The minimum norm solution is then returned.

    Examples
    --------
    Fit a noisy sum of defocus and astigmatism up to the order 4:

    .. code-block:: python

        import numpy
        from pyzernike import zernike_fit, zernike_polynomial

        rng = numpy.random.default_rng(0)
        rho = numpy.sqrt(rng.uniform(0, 1, 5000))
        theta = rng.uniform(0, 2 * numpy.pi, 5000)

        f = 0.5 * zernike_polynomial(rho, theta, n=2, m=0) - 0.2 * zernike_polynomial(rho, theta, n=2, m=2)
        f += rng.normal(0, 0.01, f.shape)

        coefficients = zernike_fit(rho, theta, f, order=4)
        # coefficients[4] ~ 0.5 (Z_2^0) and coefficients[5] ~ -0.2 (Z_2^2)

    Reconstruct the fitted function and get the residuals:

    .. code-block:: python

        from pyzernike import zernike_evaluate

        coefficients, residuals = zernike_fit(rho, theta, f, order=4, return_residuals=True)
        fitted = zernike_evaluate(coefficients, rho, theta)  # (5000,)

    """
    rho, theta, values = _as_common_floating_arrays(rho=rho, theta=theta, values=values)
    weights = _as_fit_weights(weights, shape=rho.shape)
    order = _as_order(order)
    use_recursive = _check_boolean(use_recursive, "use_recursive")
    normalize = _check_boolean(normalize, "normalize")
    return_residuals = _check_boolean(return_residuals, "return_residuals")

    n_polynomials = _core_polynomials_count(order)

    valid_mask = (
        numpy.isfinite(rho) & numpy.isfinite(theta) & numpy.isfinite(values) & numpy.isfinite(weights)
        & (rho >= 0) & (rho <= 1) & (weights > 0)
    )
    n_valid = int(valid_mask.sum())
    if n_valid < n_polynomials:
        raise ValueError(
            f"The number of valid samples ({n_valid}) must be at least the number of polynomials ({n_polynomials}) "
            f"up to the order {order}."
        )

    coefficients, fitted, rank = _core_fit(
        rho=rho[valid_mask].astype(numpy.float64),
        theta=theta[valid_mask].astype(numpy.float64),
        values=values[valid_mask].astype(numpy.float64),
        weights=weights[valid_mask].astype(numpy.float64),
        order=order,
        use_recursive=use_recursive,
        normalize=normalize,
    )
    if rank < n_polynomials:
        warnings.warn(
            f"The design matrix is rank deficient (rank {rank} < {n_polynomials}): the samples do not determine "
            "all the coefficients, the minimum norm solution is returned.",
            RuntimeWarning,
            stacklevel=2,
        )

    if not return_residuals:
        return coefficients

    residuals = numpy.full(rho.shape, numpy.nan, dtype=numpy.float64)
    residuals[valid_mask] = values[valid_mask].astype(numpy.float64) - fitted
    return coefficients, residuals