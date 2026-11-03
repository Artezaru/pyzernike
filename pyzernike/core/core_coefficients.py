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

import math
from functools import lru_cache

import numpy

from .core_evaluate import _core_evaluate
from .core_indexing import FIRST_INDEX, _core_index_to_order, _core_order_to_index, _core_polynomials_count
from .core_normalization import _core_normalization
from .core_recursive import _core_recursive_up_to_order


def _osa_normalization(order: int) -> numpy.ndarray:
    r"""
    Normalization factors of Noll of all the polynomials up to the order :math:`N`, in the OSA/ANSI ordering.

    Parameters
    ----------
    order : int
        The maximum radial order :math:`N`.

    Returns
    -------
    numpy.ndarray
        The factors :math:`N_j` with shape :math:`(P,)`.

    """
    n, m = _core_index_to_order(list(range(_core_polynomials_count(order))), indexing="OSA")
    return numpy.asarray(_core_normalization(n, m), dtype=numpy.float64)


def _core_convert_coefficients(
    coefficients: numpy.ndarray,
    input_indexing: str,
    output_indexing: str,
    input_normalized: bool,
    output_normalized: bool,
) -> numpy.ndarray:
    r"""
    Convert a vector of coefficients between two indexing conventions and between the normalized and the
    not normalized polynomials, without any check.

    .. warning::

        This method is an internal function of ``pyzernike`` that is not designed to be used by the users directly.
        Please use :func:`pyzernike.zernike_coefficients_convert`.

    The function :math:`\sum_j c_j Z_j = \sum_j c'_j N_j Z_j` is unchanged, so :math:`c'_j = c_j / N_j` from the
    not normalized to the normalized polynomials, and :math:`c_j = c'_j N_j` in the other direction.

    Length of the output:

    - Same indexing: the length of the input.
    - Output in the OSA/ANSI or Noll indexing: :math:`P = \frac{(N+1)(N+2)}{2}`, where :math:`N` is the maximum
      radial order of the input polynomials (both orderings are complete up to an order).
    - Output in the Fringe indexing: the largest Fringe index of the input polynomials (the Fringe ordering is not
      sorted by radial order).

    The coefficients of the polynomials that are not in the input are set to zero.

    Parameters
    ----------
    coefficients : numpy.ndarray (1D array)
        The coefficients in the ``input_indexing`` ordering, from its first index.

    input_indexing : str
        The canonical name of the indexing of the input: ``"OSA"``, ``"Noll"`` or ``"Fringe"``.

    output_indexing : str
        The canonical name of the indexing of the output.

    input_normalized : bool
        If True, the input coefficients are those of the normalized polynomials.

    output_normalized : bool
        If True, the output coefficients are those of the normalized polynomials.

    Returns
    -------
    numpy.ndarray
        The converted coefficients as a 1D array of ``numpy.float64``.

    """
    first_input = FIRST_INDEX[input_indexing]
    n, m = _core_index_to_order(list(range(first_input, first_input + coefficients.size)), indexing=input_indexing)

    values = coefficients.astype(numpy.float64, copy=True)
    if input_normalized != output_normalized:
        factors = numpy.asarray(_core_normalization(n, m), dtype=numpy.float64)
        values = values / factors if output_normalized else values * factors

    if output_indexing == input_indexing:
        return values

    first_output = FIRST_INDEX[output_indexing]
    positions = [j - first_output for j in _core_order_to_index(n, m, indexing=output_indexing)]
    if output_indexing == "Fringe":
        size = max(positions) + 1
    else:
        size = _core_polynomials_count(max(n))
    output = numpy.zeros(size, dtype=numpy.float64)
    output[positions] = values
    return output


def _core_rotate_coefficients(coefficients: numpy.ndarray, order: int, angle: float) -> numpy.ndarray:
    r"""
    Rotate the coefficients of a sum of Zernike polynomials, without any check.

    .. warning::

        This method is an internal function of ``pyzernike`` that is not designed to be used by the users directly.
        Please use :func:`pyzernike.zernike_coefficients_rotate`.

    The output coefficients :math:`c'` describe the function in a frame rotated by the angle :math:`\phi`:
    :math:`f'(\rho, \theta) = f(\rho, \theta + \phi)`. For each pair of polynomials :math:`(m, -m)` with
    :math:`m > 0`, with the coefficients :math:`a` of :math:`R_n^m \cos(m\theta)` and :math:`b` of
    :math:`R_n^m \sin(m\theta)`:

    .. math::

        a' = a \cos(m\phi) + b \sin(m\phi)
        \quad \text{and} \quad
        b' = b \cos(m\phi) - a \sin(m\phi)

    The coefficients with :math:`m = 0` are unchanged. The normalization factors are the same for :math:`m` and
    :math:`-m`, so the relation holds for the normalized and the not normalized polynomials.

    Parameters
    ----------
    coefficients : numpy.ndarray (1D array)
        The coefficients in the OSA/ANSI ordering, with shape :math:`(P,)` and :math:`P = \frac{(N+1)(N+2)}{2}`.

    order : int
        The maximum radial order :math:`N`.

    angle : float
        The rotation angle :math:`\phi` in radians.

    Returns
    -------
    numpy.ndarray
        The rotated coefficients with shape :math:`(P,)`.

    """
    output = coefficients.astype(numpy.float64, copy=True)
    for n in range(order + 1):
        for m in range(2 - n % 2, n + 1, 2):
            j_cos, j_sin = _core_order_to_index([n, n], [m, -m], indexing="OSA")
            a, b = coefficients[j_cos], coefficients[j_sin]
            cos_phi, sin_phi = math.cos(m * angle), math.sin(m * angle)
            output[j_cos] = a * cos_phi + b * sin_phi
            output[j_sin] = b * cos_phi - a * sin_phi
    return output


@lru_cache(maxsize=32)
def _transform_matrix(order: int, scale: float, x0: float, y0: float, angle: float) -> numpy.ndarray:
    r"""
    Matrix :math:`T` of the change of pupil for the not normalized polynomials, such that :math:`c' = T c` (cached).

    The new pupil is the disk of center :math:`(x_0, y_0)` and radius :math:`s` (in the normalized coordinates of the
    initial pupil), rotated by the angle :math:`\phi`. The point :math:`p'` of the new unit disk is the point
    :math:`q = (x_0, y_0) + s R(\phi) p'` of the initial pupil, and :math:`f'(p') = f(q)`.

    :math:`f'` is a polynomial of degree :math:`N` in :math:`p'`, so it is exactly a sum of Zernike polynomials up to
    the order :math:`N`, whose coefficients are given by the orthogonality relations:

    .. math::

        T_{ij} = N_i^2 \frac{1}{\pi} \int_0^{2\pi} \int_0^1 Z_i(p') Z_j(q(p')) \rho' \, d\rho' \, d\theta'

    The integrand is a polynomial of degree :math:`2N` in :math:`p'`, so the integral is computed exactly with a
    Gauss-Legendre quadrature of :math:`N + 2` nodes in :math:`\rho'` and a uniform quadrature of :math:`2N + 2`
    nodes in :math:`\theta'`. The polynomials :math:`Z_j(q)` are evaluated with the recurrence relation, also where
    :math:`q` is outside the initial unit disk (where the polynomials are extrapolated).

    Parameters
    ----------
    order : int
        The maximum radial order :math:`N`.

    scale : float
        The radius :math:`s > 0` of the new pupil.

    x0 : float
        The x coordinate of the center of the new pupil.

    y0 : float
        The y coordinate of the center of the new pupil.

    angle : float
        The rotation angle :math:`\phi` of the new pupil in radians.

    Returns
    -------
    numpy.ndarray
        The matrix :math:`T` with shape :math:`(P, P)`.

    """
    nodes, weights = numpy.polynomial.legendre.leggauss(order + 2)
    radial_nodes = 0.5 * (nodes + 1.0)
    radial_weights = 0.5 * weights * radial_nodes  # weight rho' d rho'
    n_angles = 2 * order + 2
    angles = 2.0 * numpy.pi * numpy.arange(n_angles) / n_angles

    rho_new, theta_new = (array.ravel() for array in numpy.meshgrid(radial_nodes, angles, indexing="ij"))
    quadrature = numpy.repeat(radial_weights, n_angles) * (2.0 / n_angles)  # (1/pi) * (2 pi / M) * w_k

    cos_phi, sin_phi = math.cos(angle), math.sin(angle)
    u, v = rho_new * numpy.cos(theta_new), rho_new * numpy.sin(theta_new)
    x = x0 + scale * (cos_phi * u - sin_phi * v)
    y = y0 + scale * (sin_phi * u + cos_phi * v)
    rho_old, theta_old = numpy.hypot(x, y), numpy.arctan2(y, x)

    Z_new = _core_recursive_up_to_order(rho=rho_new, theta=theta_new, order=order, rho_derivative=[0], theta_derivative=[0], normalize=False)[0]
    Z_old = _core_recursive_up_to_order(rho=rho_old, theta=theta_old, order=order, rho_derivative=[0], theta_derivative=[0], normalize=False)[0]

    factors = _osa_normalization(order)
    matrix = (factors[:, None] ** 2) * ((Z_new * quadrature) @ Z_old.T)
    matrix.setflags(write=False)  # cached: must not be modified
    return matrix


def _core_transform_coefficients(
    coefficients: numpy.ndarray,
    order: int,
    scale: float,
    x0: float,
    y0: float,
    angle: float,
    normalize: bool,
) -> numpy.ndarray:
    r"""
    Express a sum of Zernike polynomials in a new pupil (scaled, decentered and rotated), without any check.

    .. warning::

        This method is an internal function of ``pyzernike`` that is not designed to be used by the users directly.
        Please use :func:`pyzernike.zernike_coefficients_transform`.

    The output coefficients are :math:`c' = T c` (see :func:`_transform_matrix`). For the normalized polynomials,
    :math:`c'_{norm} = D^{-1} T D \, c_{norm}` with :math:`D = \operatorname{diag}(N_j)`.

    Parameters
    ----------
    coefficients : numpy.ndarray (1D array)
        The coefficients in the OSA/ANSI ordering, with shape :math:`(P,)` and :math:`P = \frac{(N+1)(N+2)}{2}`.

    order : int
        The maximum radial order :math:`N`.

    scale : float
        The radius :math:`s > 0` of the new pupil, in units of the radius of the initial pupil.

    x0 : float
        The x coordinate of the center of the new pupil, in units of the radius of the initial pupil.

    y0 : float
        The y coordinate of the center of the new pupil, in units of the radius of the initial pupil.

    angle : float
        The rotation angle of the new pupil in radians.

    normalize : bool
        If True, the coefficients are those of the normalized polynomials.

    Returns
    -------
    numpy.ndarray
        The coefficients in the new pupil with shape :math:`(P,)`.

    """
    matrix = _transform_matrix(order, scale, x0, y0, angle)
    if not normalize:
        return matrix @ coefficients
    factors = _osa_normalization(order)
    return (matrix @ (coefficients * factors)) / factors


def _core_rms(coefficients: numpy.ndarray, order: int, normalize: bool) -> float:
    r"""
    Root mean square (around the mean) of a sum of Zernike polynomials on the unit disk, without any check.

    .. warning::

        This method is an internal function of ``pyzernike`` that is not designed to be used by the users directly.
        Please use :func:`pyzernike.zernike_rms`.

    The polynomials are orthogonal with :math:`\frac{1}{\pi} \int Z_j^2 \, dA = \frac{1}{N_j^2}`, and only the piston
    :math:`Z_0^0` has a non-zero mean, so:

    .. math::

        \sigma = \sqrt{\sum_{j > 0} \frac{c_j^2}{N_j^2}}
        \quad \text{(or } \sqrt{\sum_{j > 0} c_j^2} \text{ for the normalized polynomials)}

    Parameters
    ----------
    coefficients : numpy.ndarray (1D array)
        The coefficients in the OSA/ANSI ordering, with shape :math:`(P,)` and :math:`P = \frac{(N+1)(N+2)}{2}`.

    order : int
        The maximum radial order :math:`N`.

    normalize : bool
        If True, the coefficients are those of the normalized polynomials.

    Returns
    -------
    float
        The root mean square :math:`\sigma`.

    """
    values = coefficients[1:] if normalize else coefficients[1:] / _osa_normalization(order)[1:]
    return float(numpy.sqrt(numpy.sum(values ** 2)))


def _core_peak_to_valley(coefficients: numpy.ndarray, order: int, normalize: bool, resolution: int) -> float:
    r"""
    Peak-to-valley of a sum of Zernike polynomials on the unit disk, evaluated on a polar grid, without any check.

    .. warning::

        This method is an internal function of ``pyzernike`` that is not designed to be used by the users directly.
        Please use :func:`pyzernike.zernike_peak_to_valley`.

    The function is evaluated with the recurrence relation on a polar grid of ``resolution`` radii in :math:`[0, 1]`
    and ``2 * resolution`` angles in :math:`[0, 2\pi)`, and the peak-to-valley is the difference between the maximum
    and the minimum values on the grid.

    Parameters
    ----------
    coefficients : numpy.ndarray (1D array)
        The coefficients in the OSA/ANSI ordering, with shape :math:`(P,)` and :math:`P = \frac{(N+1)(N+2)}{2}`.

    order : int
        The maximum radial order :math:`N`.

    normalize : bool
        If True, the coefficients are those of the normalized polynomials.

    resolution : int
        The number of radii of the polar grid.

    Returns
    -------
    float
        The peak-to-valley.

    """
    rho, theta = numpy.meshgrid(
        numpy.linspace(0.0, 1.0, resolution),
        numpy.linspace(0.0, 2.0 * numpy.pi, 2 * resolution, endpoint=False),
        indexing="ij",
    )
    values = _core_evaluate(
        rho=rho.ravel(),
        theta=theta.ravel(),
        coefficients=coefficients,
        order=order,
        rho_derivative=[0],
        theta_derivative=[0],
        use_recursive=True,
        normalize=normalize,
        precompute=True,
    )[0]
    return float(values.max() - values.min())


def _core_strehl_ratio(rms: float, wavelength: float) -> float:
    r"""
    Approximation of the Strehl ratio from the root mean square of the wavefront, without any check.

    .. warning::

        This method is an internal function of ``pyzernike`` that is not designed to be used by the users directly.
        Please use :func:`pyzernike.zernike_strehl_ratio`.

    The extended Maréchal approximation is used:

    .. math::

        S \approx \exp\left( -\left( \frac{2 \pi \sigma}{\lambda} \right)^2 \right)

    Parameters
    ----------
    rms : float
        The root mean square :math:`\sigma` of the wavefront.

    wavelength : float
        The wavelength :math:`\lambda`, in the same unit as ``rms``.

    Returns
    -------
    float
        The approximated Strehl ratio :math:`S \in (0, 1]`.

    """
    return math.exp(-((2.0 * math.pi * rms / wavelength) ** 2))