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


def _zernike_order_to_index(n: List[int], m: List[int]) -> List[int]:
    r"""
    Convert lists of Zernike orders (n, m) to their OSA/ANSI indices without any check.

    .. warning::

        This method is an internal function of ``pyzernike`` that is not designed to be used by the users directly.
        Please use :func:`pyzernike.zernike_order_to_index`.

    .. math::

        j = \frac{n(n + 2) + m}{2}

    The division is an exact integer division since :math:`n(n+2) + m` is even for valid orders.

    Parameters
    ----------
    n : List[int]
        The radial orders of the Zernike polynomials. Must be a list of Python integers.

    m : List[int]
        The azimuthal frequencies of the Zernike polynomials. Must be a list of Python integers
        with the same length as ``n``, and each pair ``(n, m)`` must be valid.

    Returns
    -------
    List[int]
        The indices of the Zernike polynomials in the OSA/ANSI ordering.

    """
    return [(n_i * (n_i + 2) + m_i) // 2 for n_i, m_i in zip(n, m)]


def _is_integer(value) -> bool:
    # bool is excluded: True is an Integral but not a Zernike order
    return isinstance(value, Integral) and not isinstance(value, bool)


def _as_integer_list(values, name: str) -> List[int]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not all(_is_integer(v) for v in values):
        raise TypeError(f"{name} must be an integer or a sequence of integers.")
    return [int(v) for v in values]


@overload
def zernike_order_to_index(n: Integral, m: Integral) -> int: ...
@overload
def zernike_order_to_index(n: Sequence[Integral], m: Sequence[Integral]) -> List[int]: ...

def zernike_order_to_index(
    n: Union[Integral, Sequence[Integral]],
    m: Union[Integral, Sequence[Integral]],
) -> Union[int, List[int]]:
    r"""
    Convert Zernike orders (n, m) to their corresponding indices in the OSA/ANSI Zernike polynomial ordering.

    .. math::

        j = \frac{n(n + 2) + m}{2}

    If :math:`n < 0`, :math:`|m| > n` or :math:`(n - m)` is odd, an error is raised.

    .. seealso::

        - :func:`pyzernike.zernike_polynomial` for computing the Zernike polynomial :math:`Z_{n}^{m}(\rho, \theta)`.
        - :func:`pyzernike.zernike_index_to_order` for converting indices back to Zernike orders.

    - ``n`` and ``m`` can be two integers, or two sequences of integers with the same length.
    - If ``n`` and ``m`` are integers, a single integer index is returned.
    - If ``n`` and ``m`` are sequences, a list of integer indices is returned.
    - Mixing an integer and a sequence is not allowed.

    The first few Zernike polynomials in the OSA/ANSI ordering are:

    +-----+-----+-----+
    |  j  |  n  |  m  |
    +=====+=====+=====+
    |  0  |  0  |  0  |
    +-----+-----+-----+
    |  1  |  1  | -1  |
    +-----+-----+-----+
    |  2  |  1  |  1  |
    +-----+-----+-----+
    |  3  |  2  | -2  |
    +-----+-----+-----+
    |  4  |  2  |  0  |
    +-----+-----+-----+
    |  5  |  2  |  2  |
    +-----+-----+-----+
    | ... | ... | ... |
    +-----+-----+-----+

    Parameters
    ----------
    n : Union[Integral, Sequence[Integral]]
        The radial order, or the sequence of radial orders, of the Zernike polynomials.

    m : Union[Integral, Sequence[Integral]]
        The azimuthal frequency, or the sequence of azimuthal frequencies, of the Zernike polynomials.
        Must be of the same kind as ``n`` (integer or sequence) and, for sequences, of the same length.

    Returns
    -------
    Union[int, List[int]]
        If ``n`` and ``m`` are integers, the index ``j`` of the Zernike polynomial.

        If ``n`` and ``m`` are sequences, the list of the indices, in the same order as ``n`` and ``m``.

    Raises
    ------
    TypeError
        If ``n`` or ``m`` is not an integer or a sequence of integers, or if an integer is mixed
        with a sequence.

    ValueError
        If ``n`` and ``m`` do not have the same length, or if a pair ``(n, m)`` is not a valid
        Zernike order (:math:`n < 0`, :math:`|m| > n` or :math:`(n - m)` odd).

    Examples
    --------
    Convert a single pair of orders:

    .. code-block:: python

        from pyzernike import zernike_order_to_index

        j = zernike_order_to_index(2, 0)
        print(j)  # Output: 4

    Convert several pairs of orders:

    .. code-block:: python

        from pyzernike import zernike_order_to_index

        n = [2, 3, 4]
        m = [0, 1, -2]
        indices = zernike_order_to_index(n, m)
        print(indices)  # Output: [4, 8, 11]

    The function is the inverse of :func:`pyzernike.zernike_index_to_order`:

    .. code-block:: python

        from pyzernike import zernike_index_to_order, zernike_order_to_index

        j = [0, 5, 12]
        print(zernike_order_to_index(*zernike_index_to_order(j)))  # Output: [0, 5, 12]

    """
    scalar_n, scalar_m = _is_integer(n), _is_integer(m)
    if scalar_n != scalar_m:
        raise TypeError("n and m must be both integers or both sequences of integers.")

    if scalar_n:
        n_list, m_list = [int(n)], [int(m)]
    else:
        n_list = _as_integer_list(n, "n")
        m_list = _as_integer_list(m, "m")
        if len(n_list) != len(m_list):
            raise ValueError("n and m must have the same length.")

    if any(n_i < 0 or abs(m_i) > n_i or (n_i - m_i) % 2 != 0 for n_i, m_i in zip(n_list, m_list)):
        raise ValueError("Invalid Zernike orders: n must be non-negative, |m| must be <= n, and (n - m) must be even.")

    indices = _zernike_order_to_index(n_list, m_list)
    return indices[0] if scalar_n else indices