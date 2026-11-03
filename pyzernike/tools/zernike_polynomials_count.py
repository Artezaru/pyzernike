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
from ..core.core_indexing import _core_polynomials_count


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
    indices ``range(P(N))``. The same holds for the Noll indices ``range(1, P(N) + 1)``, but not for the Fringe
    ordering, which is not sorted by radial order.

    :math:`P(N)` is also the length of a complete vector of coefficients up to the order :math:`N`, as expected by the
    functions working on coefficients (e.g. :func:`pyzernike.zernike_evaluate` or :func:`pyzernike.zernike_rms`).

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

    **Shape of the input and of the output.** ``order`` can be a single non-negative integer, or a sequence (list,
    tuple or 1D numpy array) of non-negative integers:

    - If ``order`` is an integer, a single integer is returned.
    - If ``order`` is a sequence, a list of integers is returned.

    .. seealso::

        - :func:`pyzernike.zernike_order_to_index` for converting Zernike orders to indices.
        - :func:`pyzernike.zernike_index_to_order` for converting indices to Zernike orders.
        - :func:`pyzernike.zernike_polynomial_up_to_order` for computing all the polynomials up to an order.
        - The page :doc:`../../mathematical_description/conventions` for the indexing conventions.

    Parameters
    ----------
    order : Union[Integral, Sequence[Integral], numpy.ndarray]
        The maximum radial order :math:`N`, or a sequence (list, tuple or 1D numpy array) of maximum radial orders.

    Returns
    -------
    Union[int, List[int]]
        If ``order`` is an integer, the number of Zernike polynomials up to this order.

        If ``order`` is a sequence, the list of the numbers of Zernike polynomials, in the same
        order as ``order``.

    Raises
    ------
    TypeError
        If ``order`` is not an integer, a sequence of integers or a 1D numpy array of integers.

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
    orders = _as_integers(order, "order")

    # Single integer
    if isinstance(orders, int):
        if orders < 0:
            raise ValueError("order must be a non-negative integer.")
        return _core_polynomials_count(order=orders)

    # Sequence of integers
    if any(N < 0 for N in orders):
        raise ValueError("order must contain non-negative integers.")
    return [_core_polynomials_count(order=N) for N in orders]