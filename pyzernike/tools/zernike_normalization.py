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
from typing import List, Sequence, Union, overload

from ..core.parsing import _as_integers
from ..core.core_normalization import _core_normalization


@overload
def zernike_normalization(n: Integral, m: Integral) -> float: ...
@overload
def zernike_normalization(n: Sequence[Integral], m: Sequence[Integral]) -> List[float]: ...

def zernike_normalization(
    n: Union[Integral, Sequence[Integral]],
    m: Union[Integral, Sequence[Integral]],
) -> Union[float, List[float]]:
    r"""
    Compute the normalization factors of Noll [1]_ of the Zernike polynomials :math:`Z_{n}^{m}`.

    The polynomials computed by ``pyzernike`` are not normalized by default (:math:`Z_{n}^{m}(1, \theta)` has an
    amplitude of 1). The normalized polynomials :math:`N_{n}^{m} Z_{n}^{m}` are orthonormal on the unit disk for the
    inner product averaged over the disk:

    .. math::

        \frac{1}{\pi} \int_{0}^{2\pi} \int_{0}^{1} N_{n}^{m} Z_{n}^{m}(\rho, \theta) \, N_{n'}^{m'} Z_{n'}^{m'}(\rho, \theta) \, \rho \, d\rho \, d\theta = \delta_{nn'} \delta_{mm'}

    with:

    .. math::

        N_{n}^{m} = \sqrt{(2 - \delta_{m0})(n + 1)} = \begin{cases}
            \sqrt{n + 1} & \text{if } m = 0 \\
            \sqrt{2(n + 1)} & \text{if } m \neq 0
        \end{cases}

    For :math:`n > 0`, the normalized polynomial has a unit root mean square over the disk, so that its coefficient in
    a decomposition is directly the RMS of the corresponding aberration. The coefficients :math:`c_j` of the not
    normalized polynomials and :math:`c'_j` of the normalized polynomials of the same function are related by
    :math:`c'_j = c_j / N_j` (see :func:`pyzernike.zernike_coefficients_convert`).

    The first factors are:

    +-----+-----+--------------------+
    |  n  |  m  |  :math:`N_{n}^{m}` |
    +=====+=====+====================+
    |  0  |  0  |  1                 |
    +-----+-----+--------------------+
    |  1  | ±1  |  2                 |
    +-----+-----+--------------------+
    |  2  |  0  |  :math:`\sqrt{3}`  |
    +-----+-----+--------------------+
    |  2  | ±2  |  :math:`\sqrt{6}`  |
    +-----+-----+--------------------+
    |  3  | ±1  |  :math:`\sqrt{8}`  |
    +-----+-----+--------------------+
    |  3  | ±3  |  :math:`\sqrt{8}`  |
    +-----+-----+--------------------+
    |  4  |  0  |  :math:`\sqrt{5}`  |
    +-----+-----+--------------------+

    **Shape of the inputs and of the output.** ``n`` and ``m`` can be two integers, or two sequences of integers
    (lists, tuples or 1D numpy arrays) with the same length:

    - If ``n`` and ``m`` are integers, a single float is returned.
    - If ``n`` and ``m`` are sequences, a list of floats is returned.
    - Mixing an integer and a sequence is not allowed.

    .. seealso::

        - :func:`pyzernike.zernike_polynomial` and :func:`pyzernike.zernike_polynomial_up_to_order` with
          ``normalize=True`` to compute the normalized polynomials directly.
        - :func:`pyzernike.zernike_coefficients_convert` to convert vectors of coefficients between the normalized and
          the not normalized polynomials.
        - :func:`pyzernike.zernike_rms` for the RMS of a sum of polynomials from its coefficients.
        - The page :doc:`../../mathematical_description/conventions` for the normalization and the orthogonality of the
          Zernike polynomials.

    Parameters
    ----------
    n : Union[Integral, Sequence[Integral], numpy.ndarray]
        The radial order, or the sequence of radial orders, of the Zernike polynomials.

    m : Union[Integral, Sequence[Integral], numpy.ndarray]
        The azimuthal frequency, or the sequence of azimuthal frequencies, of the Zernike polynomials.
        Must be of the same kind as ``n`` (integer or sequence) and, for sequences, of the same length.

    Returns
    -------
    Union[float, List[float]]
        If ``n`` and ``m`` are integers, the normalization factor :math:`N_{n}^{m}`.

        If ``n`` and ``m`` are sequences, the list of the normalization factors, in the same order as ``n`` and ``m``.

    Raises
    ------
    TypeError
        If ``n`` or ``m`` is not an integer, a sequence of integers or a 1D numpy array of integers, or if an
        integer is mixed with a sequence.

    ValueError
        If ``n`` and ``m`` do not have the same length, or if a pair ``(n, m)`` is not a valid Zernike order
        (:math:`n < 0`, :math:`|m| > n` or :math:`(n - m)` odd).

    References
    ----------
    .. [1] R. J. Noll, "Zernike polynomials and atmospheric turbulence," Journal of the Optical Society
       of America 66(3), 207-211 (1976). https://doi.org/10.1364/JOSA.66.000207

    Examples
    --------
    Normalization factor of the defocus :math:`Z_{2}^{0}`:

    .. code-block:: python

        from pyzernike import zernike_normalization

        print(zernike_normalization(2, 0))  # Output: 1.7320508075688772

    Normalize the polynomials computed without normalization:

    .. code-block:: python

        import numpy
        from pyzernike import zernike_polynomial, zernike_normalization

        rho, theta = numpy.meshgrid(numpy.linspace(0, 1, 100), numpy.linspace(0, 2 * numpy.pi, 100))
        n, m = [2, 2, 3], [-2, 0, 1]
        Z = zernike_polynomial(rho, theta, n, m, to_list=False)
        Z_normalized = Z * numpy.reshape(zernike_normalization(n, m), (-1, 1, 1))
        # same as zernike_polynomial(rho, theta, n, m, normalize=True, to_list=False)

    """
    n_values, m_values = _as_integers(n, "n"), _as_integers(m, "m")
    scalar_n, scalar_m = isinstance(n_values, int), isinstance(m_values, int)
    if scalar_n != scalar_m:
        raise TypeError("n and m must be both integers or both sequences of integers.")

    if scalar_n:
        n_list, m_list = [n_values], [m_values]
    else:
        n_list, m_list = n_values, m_values
        if len(n_list) != len(m_list):
            raise ValueError("n and m must have the same length.")

    if any(n_i < 0 or abs(m_i) > n_i or (n_i - m_i) % 2 != 0 for n_i, m_i in zip(n_list, m_list)):
        raise ValueError("Invalid Zernike orders: n must be non-negative, |m| must be <= n, and (n - m) must be even.")

    factors = _core_normalization(n=n_list, m=m_list)
    return factors[0] if scalar_n else factors