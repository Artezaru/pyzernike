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

from typing import Dict, List
import numpy
import math

from .core_indexing import _core_index_to_order, _core_order_to_index, _core_polynomials_count
from .core_normalization import _core_normalization

def _core_recursive_up_to_order(
    rho: numpy.ndarray,
    theta: numpy.ndarray,
    order: int,
    rho_derivative: List[int],
    theta_derivative: List[int],
    normalize: bool,
) -> numpy.ndarray:
    r"""
    Compute all the Zernike polynomials :math:`Z_{n}^{m}(\rho, \theta)` up to the
    radial order ``order`` using the recurrence relation of Shakibaei and Paramesran.

    The radial polynomials are built with the three-term relation:

    .. math::

        R_{n}^{m}(\rho) = \rho \, A_{n}^{m}(\rho) - R_{n-2}^{m}(\rho) \quad \text{with} \quad A_{n}^{m}(\rho) = R_{n-1}^{|m-1|}(\rho) + R_{n-1}^{m+1}(\rho)

    for :math:`n \geq 1`, :math:`0 \leq m \leq n` and :math:`(n - m)` even, with the
    initial condition :math:`R_{0}^{0}(\rho) = 1` and the convention
    :math:`R_{n}^{m}(\rho) = 0` whenever :math:`m > n`.

    Differentiating the relation :math:`k` times with the Leibniz rule
    :math:`\frac{d^k}{d\rho^k}\left[\rho f(\rho)\right] = \rho f^{(k)}(\rho) + k f^{(k-1)}(\rho)`
    gives a recurrence of the same form for each radial derivative order :math:`k \geq 1`:

    .. math::

        \frac{d^k R_{n}^{m}}{d\rho^k} = \rho \, \frac{d^k A_{n}^{m}}{d\rho^k} + k \, \frac{d^{k-1} A_{n}^{m}}{d\rho^{k-1}} - \frac{d^k R_{n-2}^{m}}{d\rho^k}

    with the diagonal terms :math:`\frac{d^k R_{n}^{n}}{d\rho^k} = \frac{n!}{(n-k)!} \rho^{n-k}`
    and :math:`\frac{d^k R_{n}^{m}}{d\rho^k} = 0` for :math:`k > n`.

    The radial orders :math:`n` are processed in increasing order and, for each :math:`n`, the
    derivative orders :math:`k = 0, \dots, \max(\text{rho\_derivatives})` in increasing order.
    Only the radial terms of the rows :math:`n-1` and :math:`n-2` are kept in memory.

    The full Zernike polynomials and their angular derivatives are then assembled as:

    .. math::

        \frac{\partial^{a+b} Z_{n}^{m}}{\partial \rho^{a} \partial \theta^{b}} = \begin{cases}
            \frac{d^a R_{n}^{m}}{d\rho^a} \, m^{b} \cos\left(m\theta + b\frac{\pi}{2}\right) & \text{if } m > 0 \\
            \frac{d^a R_{n}^{0}}{d\rho^a} \, \delta_{b,0} & \text{if } m = 0 \\
            \frac{d^a R_{n}^{|m|}}{d\rho^a} \, |m|^{b} \sin\left(|m|\theta + b\frac{\pi}{2}\right) & \text{if } m < 0
        \end{cases}

    Parameters
    -----------
    rho : numpy.ndarray (N-D array)
        The radial coordinate values with shape ``(...,)`` and floating point dtype.

    theta : numpy.ndarray (N-D array)
        The angular coordinate values with shape ``(...,)`` and floating point dtype.
        Must have same shape and dtype as ``rho``.

    order : int
        The maximum order of the Zernike polynomials to compute.
        It must be a non-negative integer.

    rho_derivative : List[int]
        The orders of the derivatives with respect to rho.

    theta_derivative : List[int]
        The orders of the derivatives with respect to theta.
        Must have the same length as ``rho_derivative``.

    normalize : bool
        If True, the Zernike polynomials (and their derivatives) are multiplied by the normalization factors of Noll
        :math:`N_{n}^{m} = \sqrt{(2 - \delta_{m0})(n + 1)}` (see :func:`pyzernike.core.core_normalization._core_normalization`).

    Returns
    -------
    numpy.ndarray
        An array of shape ``(D, P, ...)`` and dtype of ``rho``, where :math:`D` is the length of
        ``rho_derivative`` and :math:`P` the number of polynomials up to ``order``
        (see :func:`pyzernike.zernike_polynomials_count`). ``output[d, j]`` is the polynomial of
        OSA/ANSI index ``j`` for the pair ``(rho_derivative[d], theta_derivative[d])``.
        See the function :func:`pyzernike.zernike_index_to_order` for the ordering.

    References
    ----------
    .. [1] B. H. Shakibaei and R. Paramesran, "Recursive formula to compute Zernike radial
       polynomials," Optics Letters 38(14), 2487 (2013). https://doi.org/10.1364/OL.38.002487

    """
    shape = rho.shape
    dtype = rho.dtype
    n_polynomials = _core_polynomials_count(order)
    n_derivatives = len(rho_derivative)
    max_drho = max(rho_derivative) if rho_derivative else 0

    output = numpy.empty((n_derivatives, n_polynomials) + shape, dtype=dtype)
    if n_derivatives == 0:
        return output

    zero = numpy.zeros(shape, dtype=dtype)
    zero.flags.writeable = False
    buffer = numpy.empty(shape, dtype=dtype)

    # rows[n][drho][m // 2] = d^drho R_n^m / drho^drho, only rows n - 1 and n - 2 are kept
    rows: Dict[int, List[List[numpy.ndarray]]] = {}

    def _get(n: int, m: int, drho: int) -> numpy.ndarray:
        if n < 0 or m > n or drho > n:
            return zero
        return rows[n][drho][m // 2]

    frequencies = numpy.arange(1, order + 1, dtype=dtype).reshape((-1,) + (1,) * theta.ndim)
    angles = frequencies * theta
    tables = {"cos": numpy.cos(angles), "sin": numpy.sin(angles, out=angles)}

    COSINE_DERIVATIVE = {0: (1.0, "cos"), 1: (-1.0, "sin"), 2: (-1.0, "cos"), 3: (1.0, "sin")}
    SINE_DERIVATIVE = {0: (1.0, "sin"), 1: (1.0, "cos"), 2: (-1.0, "sin"), 3: (-1.0, "cos")}

    for n in range(order + 1):
        row: List[List[numpy.ndarray]] = []

        for drho in range(max_drho + 1):
            row_drho: List[numpy.ndarray] = []

            for m in range(n % 2, n + 1, 2):
                if drho > n:
                    radial = zero

                elif n == m:
                    radial = numpy.ones(shape, dtype=dtype) if n == 0 else math.perm(n, drho) * numpy.power(rho, n - drho)

                else:
                    radial = numpy.add(_get(n - 1, abs(m - 1), drho), _get(n - 1, m + 1, drho))
                    radial *= rho
                    if drho > 0:
                        numpy.add(_get(n - 1, abs(m - 1), drho - 1), _get(n - 1, m + 1, drho - 1), out=buffer)
                        buffer *= drho
                        radial += buffer
                    radial -= _get(n - 2, m, drho)

                row_drho.append(radial)
            row.append(row_drho)

        rows[n] = row
        rows.pop(n - 2, None)

        # OSA/ANSI indices of the polynomials of the row n
        row_m = list(range(-n, n + 1, 2))
        row_index = dict(zip(row_m, _core_order_to_index([n] * len(row_m), row_m)))

        for index_derivative, (drho, dtheta) in enumerate(zip(rho_derivative, theta_derivative)):
            for m in range(n % 2, n + 1, 2):
                radial = row[drho][m // 2]

                if m == 0:
                    output[index_derivative, row_index[0]] = radial if dtheta == 0 else 0.0
                    continue

                for sign_m, derivative in ((1, COSINE_DERIVATIVE), (-1, SINE_DERIVATIVE)):
                    sign, func = derivative[dtheta % 4]
                    out = output[index_derivative, row_index[sign_m * m]]
                    numpy.multiply(radial, tables[func][m - 1], out=out)
                    factor = sign * float(m) ** dtheta
                    if factor != 1.0:
                        out *= factor

    if normalize:
        n_list, m_list = _core_index_to_order(list(range(n_polynomials)))
        factors = numpy.asarray(_core_normalization(n_list, m_list), dtype=dtype)
        output *= factors.reshape((1, n_polynomials) + (1,) * len(shape))

    return output