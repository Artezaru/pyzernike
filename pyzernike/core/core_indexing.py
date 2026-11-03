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

r"""
Conversions between the Zernike orders :math:`(n, m)` and the indices of the OSA/ANSI, Noll and Fringe orderings.

These functions do not check their inputs (see :mod:`pyzernike.core.parsing` and the public functions
:func:`pyzernike.zernike_index_to_order` and :func:`pyzernike.zernike_order_to_index`). They use exact integer
arithmetic (``math.isqrt``), so that the results are exact for any index.
"""

from math import isqrt
from typing import List, Tuple


# First index of each indexing convention
FIRST_INDEX = {"OSA": 0, "Noll": 1, "Fringe": 1}


def _osa_index_to_order(j: int) -> Tuple[int, int]:
    r"""
    Convert an OSA/ANSI index :math:`j \geq 0` to the Zernike orders :math:`(n, m)`.

    .. math::

        n = \left\lfloor \frac{\lfloor \sqrt{8j + 1} \rfloor - 1}{2} \right\rfloor \quad \text{and} \quad m = 2j - n(n + 2)

    Parameters
    ----------
    j : int
        The OSA/ANSI index.

    Returns
    -------
    Tuple[int, int]
        The orders ``(n, m)``.

    """
    n = (isqrt(8 * j + 1) - 1) // 2
    return n, 2 * j - n * (n + 2)


def _noll_index_to_order(j: int) -> Tuple[int, int]:
    r"""
    Convert a Noll index :math:`j \geq 1` to the Zernike orders :math:`(n, m)`.

    The radial order :math:`n` is the largest integer such that :math:`\frac{n(n+1)}{2} < j`. In the row :math:`n`, with
    :math:`r = j - \frac{n(n+1)}{2} \in [1, n + 1]`, the polynomials are sorted by increasing :math:`|m|`:

    .. math::

        |m| = \begin{cases}
            2 \left\lfloor \frac{r}{2} \right\rfloor & \text{if } n \text{ is even} \\
            2 \left\lfloor \frac{r - 1}{2} \right\rfloor + 1 & \text{if } n \text{ is odd}
        \end{cases}

    and :math:`m > 0` (cosine) for the even indices, :math:`m < 0` (sine) for the odd indices.

    Parameters
    ----------
    j : int
        The Noll index.

    Returns
    -------
    Tuple[int, int]
        The orders ``(n, m)``.

    """
    n = (isqrt(8 * (j - 1) + 1) - 1) // 2
    r = j - n * (n + 1) // 2
    abs_m = 2 * (r // 2) if n % 2 == 0 else 2 * ((r - 1) // 2) + 1
    return n, (abs_m if (abs_m == 0 or j % 2 == 0) else -abs_m)


def _fringe_index_to_order(j: int) -> Tuple[int, int]:
    r"""
    Convert a Fringe (University of Arizona) index :math:`j \geq 1` to the Zernike orders :math:`(n, m)`.

    With :math:`p = \frac{n + |m|}{2}`, the indices :math:`p^2 + 1 \leq j \leq (p + 1)^2` correspond to the same :math:`p`,
    so :math:`p = \lfloor \sqrt{j - 1} \rfloor`. With :math:`d = (p + 1)^2 - j = 2|m| - [m < 0]`:

    .. math::

        |m| = \left\lfloor \frac{d + 1}{2} \right\rfloor, \quad m < 0 \text{ if } d \text{ is odd}, \quad n = 2p - |m|

    Parameters
    ----------
    j : int
        The Fringe index.

    Returns
    -------
    Tuple[int, int]
        The orders ``(n, m)``.

    """
    p = isqrt(j - 1)
    d = (p + 1) ** 2 - j
    abs_m = (d + 1) // 2
    return 2 * p - abs_m, (-abs_m if d % 2 else abs_m)


def _osa_order_to_index(n: int, m: int) -> int:
    r"""
    Convert valid Zernike orders :math:`(n, m)` to their OSA/ANSI index (exact integer division):

    .. math::

        j = \frac{n(n + 2) + m}{2}

    Parameters
    ----------
    n : int
        The radial order.

    m : int
        The azimuthal frequency.

    Returns
    -------
    int
        The OSA/ANSI index.

    """
    return (n * (n + 2) + m) // 2


def _noll_order_to_index(n: int, m: int) -> int:
    r"""
    Convert valid Zernike orders :math:`(n, m)` to their Noll index:

    .. math::

        j = \frac{n(n + 1)}{2} + |m| + \begin{cases}
            0 & \text{if } m > 0 \text{ and } n \bmod 4 \in \{0, 1\}, \text{ or } m < 0 \text{ and } n \bmod 4 \in \{2, 3\} \\
            1 & \text{otherwise (in particular if } m = 0 \text{)}
        \end{cases}

    Parameters
    ----------
    n : int
        The radial order.

    m : int
        The azimuthal frequency.

    Returns
    -------
    int
        The Noll index.

    """
    shift = 0 if (m > 0 and n % 4 in (0, 1)) or (m < 0 and n % 4 in (2, 3)) else 1
    return n * (n + 1) // 2 + abs(m) + shift


def _fringe_order_to_index(n: int, m: int) -> int:
    r"""
    Convert valid Zernike orders :math:`(n, m)` to their Fringe (University of Arizona) index:

    .. math::

        j = \left(1 + \frac{n + |m|}{2}\right)^2 - 2|m| + [m < 0]

    Parameters
    ----------
    n : int
        The radial order.

    m : int
        The azimuthal frequency.

    Returns
    -------
    int
        The Fringe index.

    """
    return (1 + (n + abs(m)) // 2) ** 2 - 2 * abs(m) + (1 if m < 0 else 0)


_INDEX_TO_ORDER = {"OSA": _osa_index_to_order, "Noll": _noll_index_to_order, "Fringe": _fringe_index_to_order}
_ORDER_TO_INDEX = {"OSA": _osa_order_to_index, "Noll": _noll_order_to_index, "Fringe": _fringe_order_to_index}


def _core_index_to_order(j: List[int], indexing: str = "OSA") -> Tuple[List[int], List[int]]:
    r"""
    Convert a list of indices to their Zernike orders (n, m) without any check.

    .. warning::

        This method is an internal function of ``pyzernike`` that is not designed to be used by the users directly.
        Please use :func:`pyzernike.zernike_index_to_order`.

    Parameters
    ----------
    j : List[int]
        The indices of the Zernike polynomials. Must be a list of Python integers, greater than or equal to the
        first index of the convention (0 for OSA/ANSI, 1 for Noll and Fringe).

    indexing : str, optional
        The canonical name of the indexing convention: ``"OSA"``, ``"Noll"`` or ``"Fringe"``. Default is ``"OSA"``.

    Returns
    -------
    List[int]
        The radial orders (n) of the Zernike polynomials.

    List[int]
        The azimuthal frequencies (m) of the Zernike polynomials.

    """
    convert = _INDEX_TO_ORDER[indexing]
    orders = [convert(i) for i in j]
    return [n for n, _ in orders], [m for _, m in orders]


def _core_order_to_index(n: List[int], m: List[int], indexing: str = "OSA") -> List[int]:
    r"""
    Convert lists of Zernike orders (n, m) to their indices without any check.

    .. warning::

        This method is an internal function of ``pyzernike`` that is not designed to be used by the users directly.
        Please use :func:`pyzernike.zernike_order_to_index`.

    Parameters
    ----------
    n : List[int]
        The radial orders of the Zernike polynomials. Must be a list of Python integers.

    m : List[int]
        The azimuthal frequencies of the Zernike polynomials. Must be a list of Python integers
        with the same length as ``n``, and each pair ``(n, m)`` must be valid.

    indexing : str, optional
        The canonical name of the indexing convention: ``"OSA"``, ``"Noll"`` or ``"Fringe"``. Default is ``"OSA"``.

    Returns
    -------
    List[int]
        The indices of the Zernike polynomials in the given convention.

    """
    convert = _ORDER_TO_INDEX[indexing]
    return [convert(n_i, m_i) for n_i, m_i in zip(n, m)]


def _core_polynomials_count(order: int) -> int:
    r"""
    Return the number of Zernike polynomials with :math:`0 \leq n \leq N`, without any check:

    .. math::

        P = \frac{(N + 1)(N + 2)}{2}

    which is also the OSA/ANSI index of :math:`Z_{N}^{N}` plus one.

    Parameters
    ----------
    order : int
        The maximum radial order :math:`N \geq 0`.

    Returns
    -------
    int
        The number of polynomials.

    """
    return (order + 1) * (order + 2) // 2