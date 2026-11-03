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

from typing import Callable, List

import numpy

from .core_indexing import _core_index_to_order, _core_polynomials_count
from .core_polynomial import _core_polynomials
from .core_recursive import _core_recursive_up_to_order
from .core_xy_polynomial import _core_xy_polynomials
from .core_xy_recursive import _core_xy_recursive_up_to_order

# Maximum number of values D * P * K of the polynomials kept in memory at once (32 MiB in double precision)
CHUNK_ELEMENTS = 2 ** 22


def _evaluate_by_chunks(
    rho: numpy.ndarray,
    coefficients: numpy.ndarray,
    n_derivatives: int,
    compute: Callable[[slice], numpy.ndarray],
) -> numpy.ndarray:
    r"""
    Sum the polynomials weighted by the coefficients, by chunks of points.

    Parameters
    ----------
    rho : numpy.ndarray (1D array)
        The radial coordinates of the points (only used for the number of points and the dtype).

    coefficients : numpy.ndarray (1D array)
        The coefficients :math:`c_j` with shape :math:`(P,)`.

    n_derivatives : int
        The number :math:`D` of derivatives.

    compute : Callable[[slice], numpy.ndarray]
        A function that returns the polynomials of shape :math:`(D, P, K)` for the points ``rho[chunk]``.

    Returns
    -------
    numpy.ndarray
        The sums :math:`\sum_j c_j Z_j` with shape :math:`(D, K)` and the dtype of ``rho``.

    """
    n_points = rho.size
    chunk_size = max(1, CHUNK_ELEMENTS // max(1, n_derivatives * coefficients.size))
    weights = coefficients.astype(rho.dtype)
    output = numpy.empty((n_derivatives, n_points), dtype=rho.dtype)
    for start in range(0, n_points, chunk_size):
        chunk = slice(start, min(start + chunk_size, n_points))
        output[:, chunk] = numpy.tensordot(weights, compute(chunk), axes=(0, 1))
    return output


def _core_evaluate(
    rho: numpy.ndarray,
    theta: numpy.ndarray,
    coefficients: numpy.ndarray,
    order: int,
    rho_derivative: List[int],
    theta_derivative: List[int],
    use_recursive: bool,
    normalize: bool,
    precompute: bool,
) -> numpy.ndarray:
    r"""
    Evaluate a sum of Zernike polynomials :math:`\sum_j c_j Z_j(\rho, \theta)` and its derivatives without any check.

    .. warning::

        This method is an internal function of ``pyzernike`` that is not designed to be used by the users directly.
        Please use :func:`pyzernike.zernike_evaluate`.

    The polynomials are computed by chunks of points (at most ``CHUNK_ELEMENTS`` values at once), so that the
    :math:`P` arrays of the polynomials are never stored for all the points.

    Parameters
    ----------
    rho : numpy.ndarray (1D array)
        The radial coordinates :math:`\rho \in [0, 1]`, with floating point dtype.

    theta : numpy.ndarray (1D array)
        The angular coordinates :math:`\theta`, with the same shape and dtype as ``rho``.

    coefficients : numpy.ndarray (1D array)
        The coefficients :math:`c_j` in the OSA/ANSI ordering, with shape :math:`(P,)` and
        :math:`P = \frac{(N+1)(N+2)}{2}`.

    order : int
        The maximum radial order :math:`N`.

    rho_derivative : List[int]
        The orders of the derivatives with respect to :math:`\rho`.

    theta_derivative : List[int]
        The orders of the derivatives with respect to :math:`\theta`, with the same length as ``rho_derivative``.

    use_recursive : bool
        If True, the polynomials are computed with the recurrence relation, otherwise with the explicit sum.

    normalize : bool
        If True, the coefficients are those of the polynomials normalized by the factors of Noll.

    precompute : bool
        Only used if ``use_recursive`` is False. If True, the powers of :math:`\rho` and the cosine and sine terms
        are cached and shared between the polynomials of a chunk.

    Returns
    -------
    numpy.ndarray
        An array of shape :math:`(D, K)` and dtype of ``rho``, where :math:`D` is the length of ``rho_derivative``
        and :math:`K` the number of points.

    """
    n_derivatives = len(rho_derivative)
    n_polynomials = _core_polynomials_count(order)
    n, m = _core_index_to_order(list(range(n_polynomials)), indexing="OSA")

    def compute(chunk: slice) -> numpy.ndarray:
        if use_recursive:
            return _core_recursive_up_to_order(
                rho=rho[chunk],
                theta=theta[chunk],
                order=order,
                rho_derivative=rho_derivative,
                theta_derivative=theta_derivative,
                normalize=normalize,
            )
        return _core_polynomials(
            rho=rho[chunk],
            theta=theta[chunk],
            n=n * n_derivatives,
            m=m * n_derivatives,
            rho_derivative=[a for a in rho_derivative for _ in range(n_polynomials)],
            theta_derivative=[b for b in theta_derivative for _ in range(n_polynomials)],
            flag_radial=False,
            precompute=precompute,
            normalize=normalize,
        ).reshape((n_derivatives, n_polynomials, -1))

    return _evaluate_by_chunks(rho=rho, coefficients=coefficients, n_derivatives=n_derivatives, compute=compute)


def _core_xy_evaluate(
    rho: numpy.ndarray,
    theta: numpy.ndarray,
    coefficients: numpy.ndarray,
    order: int,
    x_derivative: List[int],
    y_derivative: List[int],
    Rx: float,
    Ry: float,
    alpha: float,
    h: float,
    use_recursive: bool,
    normalize: bool,
    precompute: bool,
) -> numpy.ndarray:
    r"""
    Evaluate a sum of Zernike polynomials on an elliptic annulus :math:`\sum_j c_j Zxy_j(x, y)` and its derivatives
    with respect to :math:`x` and :math:`y`, without any check.

    .. warning::

        This method is an internal function of ``pyzernike`` that is not designed to be used by the users directly.
        Please use :func:`pyzernike.xy_zernike_evaluate`.

    The polynomials are computed by chunks of points (at most ``CHUNK_ELEMENTS`` values at once), so that the
    :math:`P` arrays of the polynomials are never stored for all the points.

    Parameters
    ----------
    rho : numpy.ndarray (1D array)
        The equivalent radial coordinates :math:`\rho_{eq} \in [0, 1]` of the points, with floating point dtype.

    theta : numpy.ndarray (1D array)
        The equivalent angular coordinates :math:`\theta_{eq}`, with the same shape and dtype as ``rho``.

    coefficients : numpy.ndarray (1D array)
        The coefficients :math:`c_j` in the OSA/ANSI ordering, with shape :math:`(P,)` and
        :math:`P = \frac{(N+1)(N+2)}{2}`.

    order : int
        The maximum radial order :math:`N`.

    x_derivative : List[int]
        The orders of the derivatives with respect to :math:`x`.

    y_derivative : List[int]
        The orders of the derivatives with respect to :math:`y`, with the same length as ``x_derivative``.

    Rx : float
        The semi-axis :math:`R_x` of the ellipse along its x axis.

    Ry : float
        The semi-axis :math:`R_y` of the ellipse along its y axis.

    alpha : float
        The rotation angle :math:`\alpha` of the ellipse in radians.

    h : float
        The ratio :math:`h` of the inner to the outer semi-axes, in :math:`[0, 1)`.

    use_recursive : bool
        If True, the polynomials are computed with the recurrence relation, otherwise with the explicit sums.

    normalize : bool
        If True, the coefficients are those of the polynomials normalized by the factors of Noll.

    precompute : bool
        Only used if ``use_recursive`` is False. If True, the powers and the cosine and sine terms are cached and
        shared between the polynomials of a chunk.

    Returns
    -------
    numpy.ndarray
        An array of shape :math:`(D, K)` and dtype of ``rho``, where :math:`D` is the length of ``x_derivative``
        and :math:`K` the number of points.

    """
    n_derivatives = len(x_derivative)
    n_polynomials = _core_polynomials_count(order)
    n, m = _core_index_to_order(list(range(n_polynomials)), indexing="OSA")

    def compute(chunk: slice) -> numpy.ndarray:
        if use_recursive:
            return _core_xy_recursive_up_to_order(
                rho=rho[chunk],
                theta=theta[chunk],
                order=order,
                x_derivative=x_derivative,
                y_derivative=y_derivative,
                Rx=Rx,
                Ry=Ry,
                alpha=alpha,
                h=h,
                normalize=normalize,
            )
        return _core_xy_polynomials(
            rho=rho[chunk],
            theta=theta[chunk],
            n=n * n_derivatives,
            m=m * n_derivatives,
            x_derivative=[dx for dx in x_derivative for _ in range(n_polynomials)],
            y_derivative=[dy for dy in y_derivative for _ in range(n_polynomials)],
            Rx=Rx,
            Ry=Ry,
            alpha=alpha,
            h=h,
            precompute=precompute,
            normalize=normalize,
        ).reshape((n_derivatives, n_polynomials, -1))

    return _evaluate_by_chunks(rho=rho, coefficients=coefficients, n_derivatives=n_derivatives, compute=compute)