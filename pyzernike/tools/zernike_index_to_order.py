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

from math import isqrt
from numbers import Integral
from typing import List, Sequence, Tuple, Union, overload


def _zernike_index_to_order(j: List[int]) -> Tuple[List[int], List[int]]:
    r"""
    Convert a list of OSA/ANSI indices to their Zernike orders (n, m) without any check.

    .. warning::

        This method is an internal function of ``pyzernike`` that is not designed to be used by the users directly.
        Please use :func:`pyzernike.zernike_index_to_order`.

    The radial order is computed with the exact integer square root, so that the result is
    exact for any index (no floating point rounding for large ``j``):

    .. math::

        n = \left\lfloor \frac{\lfloor \sqrt{8j + 1} \rfloor - 1}{2} \right\rfloor \quad \text{and} \quad m = 2j - n(n + 2)

    Parameters
    ----------
    j : List[int]
        The indices of the Zernike polynomials in the OSA/ANSI ordering. Must be a list of
        non-negative Python integers.

    Returns
    -------
    List[int]
        The radial orders (n) of the Zernike polynomials.

    List[int]
        The azimuthal frequencies (m) of the Zernike polynomials.

    """
    n = [(isqrt(8 * i + 1) - 1) // 2 for i in j]
    m = [2 * i - n_i * (n_i + 2) for n_i, i in zip(n, j)]
    return n, m


@overload
def zernike_index_to_order(j: Integral) -> Tuple[int, int]: ...
@overload
def zernike_index_to_order(j: Sequence[Integral]) -> Tuple[List[int], List[int]]: ...

def zernike_index_to_order(
    j: Union[Integral, Sequence[Integral]]
) -> Union[Tuple[int, int], Tuple[List[int], List[int]]]:
    r"""
    Convert indices in the OSA/ANSI Zernike polynomial ordering to their corresponding Zernike orders (n, m).

    .. math::

        j = \frac{n(n + 2) + m}{2}

    .. seealso::

        - :func:`pyzernike.zernike_polynomial` for computing the Zernike polynomial :math:`Z_{n}^{m}(\rho, \theta)`.
        - :func:`pyzernike.zernike_order_to_index` for converting Zernike orders to indices.

    - ``j`` can be a single non-negative integer, or a sequence of non-negative integers.
    - If ``j`` is a single integer, a tuple ``(n, m)`` of integers is returned.
    - If ``j`` is a sequence, a tuple ``(n, m)`` of lists of integers is returned.

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

    The process to compute the Zernike orders from the index is as follows. Since
    :math:`-n \leq m \leq n`:

    .. math::

        n(n+2) = 2j - m \in [2j - n, 2j + n]

    So :math:`n` is the largest integer such that :math:`n(n+2)/2 \leq j`, i.e.:

    .. math::

        n = \left\lfloor \frac{\lfloor \sqrt{8j + 1} \rfloor - 1}{2} \right\rfloor \quad \text{and} \quad m = 2j - n(n + 2)

    The integer square root is used so that the result is exact for any index.

    Parameters
    ----------
    j : Union[Integral, Sequence[Integral]]
        The index, or the sequence of indices, of the Zernike polynomials in the OSA/ANSI ordering.

    Returns
    -------
    Union[Tuple[int, int], Tuple[List[int], List[int]]]
        If ``j`` is an integer, the tuple ``(n, m)`` with the radial order ``n`` and the
        azimuthal frequency ``m`` of the Zernike polynomial.

        If ``j`` is a sequence, the tuple ``(n, m)`` where ``n`` is the list of the radial
        orders and ``m`` the list of the azimuthal frequencies, in the same order as ``j``.

    Raises
    ------
    TypeError
        If ``j`` is not an integer or a sequence of integers.

    ValueError
        If any index in ``j`` is negative.

    Examples
    --------
    Convert a single index:

    .. code-block:: python

        from pyzernike import zernike_index_to_order

        n, m = zernike_index_to_order(4)
        print(n, m)  # Output: 2 0

    Convert several indices:

    .. code-block:: python

        from pyzernike import zernike_index_to_order

        j = [2, 3, 4]
        n, m = zernike_index_to_order(j)
        print(n)  # Output: [1, 2, 2]
        print(m)  # Output: [1, -2, 0]

    """
    # Single integer (bool is excluded: True is an Integral but not an index)
    if isinstance(j, Integral) and not isinstance(j, bool):
        if j < 0:
            raise ValueError("j must be a non-negative integer.")
        n, m = _zernike_index_to_order([int(j)])
        return n[0], m[0]

    # Sequence of integers
    if isinstance(j, (str, bytes)) or not isinstance(j, Sequence) or not all(isinstance(i, Integral) and not isinstance(i, bool) for i in j):
        raise TypeError("j must be an integer or a sequence of integers.")
    if any(i < 0 for i in j):
        raise ValueError("j must contain non-negative integers.")

    return _zernike_index_to_order([int(i) for i in j])