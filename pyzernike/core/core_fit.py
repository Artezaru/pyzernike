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

from typing import Tuple

import numpy

from .core_indexing import _core_index_to_order, _core_polynomials_count
from .core_polynomial import _core_polynomials
from .core_recursive import _core_recursive_up_to_order


def _core_fit(
    rho: numpy.ndarray,
    theta: numpy.ndarray,
    values: numpy.ndarray,
    weights: numpy.ndarray,
    order: int,
    use_recursive: bool,
    normalize: bool,
) -> Tuple[numpy.ndarray, numpy.ndarray, int]:
    r"""
    Fit sampled values with all the Zernike polynomials up to the order :math:`N` by weighted linear least squares,
    without any check.

    .. warning::

        This method is an internal function of ``pyzernike`` that is not designed to be used by the users directly.
        Please use :func:`pyzernike.zernike_fit` or :func:`pyzernike.xy_zernike_fit`.

    The coefficients :math:`c_j` minimize :math:`\sum_k w_k \left( f_k - \sum_j c_j Z_j(\rho_k, \theta_k) \right)^2`.
    The problem is solved with an orthogonal decomposition of the design matrix
    :math:`A_{kj} = \sqrt{w_k} Z_j(\rho_k, \theta_k)` (:func:`numpy.linalg.lstsq`), which is more stable than
    the normal equations.

    Parameters
    ----------
    rho : numpy.ndarray (1D array)
        The radial coordinates :math:`\rho_k \in [0, 1]` of the valid samples, as ``numpy.float64``.

    theta : numpy.ndarray (1D array)
        The angular coordinates :math:`\theta_k` of the valid samples, with the same shape as ``rho``.

    values : numpy.ndarray (1D array)
        The finite sampled values :math:`f_k`, with the same shape as ``rho``.

    weights : numpy.ndarray (1D array)
        The strictly positive weights :math:`w_k`, with the same shape as ``rho``.

    order : int
        The maximum radial order :math:`N \geq 0`.

    use_recursive : bool
        If True, the polynomials are computed with the recurrence relation, otherwise with the explicit sum.

    normalize : bool
        If True, the polynomials are normalized by the factors of Noll.

    Returns
    -------
    numpy.ndarray
        The coefficients :math:`c_j` with shape :math:`(P,)`, in the OSA/ANSI ordering.

    numpy.ndarray
        The fitted values :math:`\sum_j c_j Z_j(\rho_k, \theta_k)` with the shape of ``rho``.

    int
        The rank of the design matrix (lower than :math:`P` if the samples do not determine all the coefficients).

    """
    n_polynomials = _core_polynomials_count(order)

    if use_recursive:
        design = _core_recursive_up_to_order(
            rho=rho,
            theta=theta,
            order=order,
            rho_derivative=[0],
            theta_derivative=[0],
            normalize=normalize,
        )[0]
    else:
        n, m = _core_index_to_order(list(range(n_polynomials)), indexing="OSA")
        design = _core_polynomials(
            rho=rho,
            theta=theta,
            n=n,
            m=m,
            rho_derivative=[0] * n_polynomials,
            theta_derivative=[0] * n_polynomials,
            flag_radial=False,
            precompute=True,
            normalize=normalize,
        )
    design = design.T  # (K, P)

    sqrt_weights = numpy.sqrt(weights)
    coefficients, _, rank, _ = numpy.linalg.lstsq(design * sqrt_weights[:, None], values * sqrt_weights, rcond=None)
    return coefficients, design @ coefficients, int(rank)