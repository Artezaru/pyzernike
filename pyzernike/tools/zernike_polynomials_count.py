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


def _zernike_polynomials_count(order: List[int]) -> List[int]:
    r"""
    Compute the number of Zernike polynomials up to each given radial order without any check.

    .. warning::

        This method is an internal function of ``pyzernike`` that is not designed to be used by the users directly.
        Please use :func:`pyzernike.zernike_polynomials_count`.

    .. math::

        P(N) = j(N, N) + 1 = \frac{(N + 1)(N + 2)}{2}

    Parameters
    ----------
    order : List[int]
        The maximum radial orders. Must be a list of non-negative Python integers.

    Returns
    -------
    List[int]
        The number of Zernike polynomials with :math:`0 \leq n \leq N` for each order :math:`N`.

    """
    return [(N + 1) * (N + 2) // 2 for N in order]


def _is_integer(value) -> bool:
    # bool is excluded: True is an Integral but not a radial order
    return isinstance(value, Integral) and not isinstance(value, bool)


@overload
def zernike_polynomials_count(order: Integral) -> int: ...
@overload
def zernike_polynomials_count(order: Sequence[Integral]) -> List[int]: ...

def zernike_polynomials_count(
    order: Union[Integral, Sequence[Integral]]
) -> Union[int, List[int]]:
    r"""
    Compute the number of Zernike polynomials :math:`Z_{n}^{m}` with a radial order
    :math:`0 \leq n \leq N`, i.e. the number of polynomials computed up to the order :math:`N`.

    For each radial order :math:`n` there are :math:`n + 1` polynomials
    (:math:`m = -n, -n + 2, \dots, n`), so that:

    .. math::

        P(N) = \sum_{n=0}^{N} (n + 1) = \frac{(N + 1)(N + 2)}{2}

    In the OSA/ANSI ordering, the last polynomial of order :math:`N` is :math:`Z_{N}^{N}`, with
    index :math:`j(N, N) = N(N + 3)/2`. Since the indices start at 0, the number of polynomials
    is :math:`P(N) = j(N, N) + 1`, and the polynomials up to the order :math:`N` are exactly the
    indices ``range(P(N))``.

    .. seealso::

        - :func:`pyzernike.zernike_order_to_index` for converting Zernike orders to indices.
        - :func:`pyzernike.zernike_index_to_order` for converting indices to Zernike orders.
        - :func:`pyzernike.zernike_polynomial_up_to_order` for computing all the polynomials up to an order.

    - ``order`` can be a single non-negative integer, or a sequence of non-negative integers.
    - If ``order`` is an integer, a single integer is returned.
    - If ``order`` is a sequence, a list of integers is returned.

    The first values are:

    +-------+-----------+-----------+
    |  N    |  j(N, N)  |  P(N)     |
    +=======+===========+===========+
    |  0    |  0        |  1        |
    +-------+-----------+-----------+
    |  1    |  2        |  3        |
    +-------+-----------+-----------+
    |  2    |  5        |  6        |
    +-------+-----------+-----------+
    |  3    |  9        |  10       |
    +-------+-----------+-----------+
    |  4    |  14       |  15       |
    +-------+-----------+-----------+
    |  ...  |  ...      |  ...      |
    +-------+-----------+-----------+

    Parameters
    ----------
    order : Union[Integral, Sequence[Integral]]
        The maximum radial order :math:`N`, or a sequence of maximum radial orders.

    Returns
    -------
    Union[int, List[int]]
        If ``order`` is an integer, the number of Zernike polynomials up to this order.

        If ``order`` is a sequence, the list of the numbers of Zernike polynomials, in the same
        order as ``order``.

    Raises
    ------
    TypeError
        If ``order`` is not an integer or a sequence of integers.

    ValueError
        If any order is negative.

    Examples
    --------
    Number of polynomials up to the order 4:

    .. code-block:: python

        from pyzernike import zernike_polynomials_count

        P = zernike_polynomials_count(4)
        print(P)  # Output: 15

    List all the orders (n, m) up to the order 2:

    .. code-block:: python

        from pyzernike import zernike_polynomials_count, zernike_index_to_order

        n, m = zernike_index_to_order(range(zernike_polynomials_count(2)))
        print(n)  # Output: [0, 1, 1, 2, 2, 2]
        print(m)  # Output: [0, -1, 1, -2, 0, 2]

    """
    # Single integer
    if _is_integer(order):
        if order < 0:
            raise ValueError("order must be a non-negative integer.")
        return _zernike_polynomials_count([int(order)])[0]

    # Sequence of integers
    if isinstance(order, (str, bytes)) or not isinstance(order, Sequence) or not all(_is_integer(N) for N in order):
        raise TypeError("order must be an integer or a sequence of integers.")
    if any(N < 0 for N in order):
        raise ValueError("order must contain non-negative integers.")

    return _zernike_polynomials_count([int(N) for N in order])