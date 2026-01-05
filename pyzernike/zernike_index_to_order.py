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

from typing import Sequence, List, Tuple
from numbers import Integral


def zernike_index_to_order(j: Sequence[Integral]) -> Tuple[List[int], List[int]]:
    r"""
    Convert indices in the OSA/ANSI Zernike polynomial ordering to their corresponding Zernike orders (n, m).

    .. math::

        j = \frac{n(n + 2) + m}{2}

    .. seealso::

        - :func:`pyzernike.zernike_polynomial` for computing the Zernike polynomial :math:`Z_{n}^{m}(\rho, \theta)`.
        - :func:`pyzernike.zernike_order_to_index` for converting Zernike orders to indices.    

    - ``j`` must be a sequence of non-negative integers.

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

    The process to compute the Zernike orders from the index is as follows:

    .. math::

        n(n+2) = 2j - m \in [2j - n, 2j + n]

    So :

    .. math::
    
        n = \text{int}\left(\frac{-1 + \sqrt{1 + 8j}}{2}\right) \quad \text{and} \quad m = 2j - n(n + 2)

    Parameters
    ----------
    j : Sequence[Integral]
        The indices of the Zernike polynomials in the OSA/ANSI ordering.

    Returns
    -------
    List[int]
        A list of radial orders (n) of the Zernike polynomials.

    List[int]
        A list of azimuthal frequencies (m) of the Zernike polynomials.

    Raises
    ------
    TypeError
        If `j` is not a sequence of integers.

    Examples
    --------

    .. code-block:: python

        from pyzernike import zernike_index_to_order

        j = [2, 3, 4]
        n, m = zernike_index_to_order(j)
        print(n)  # Output: [1, 2, 2]
        print(m)  # Output: [1, -2, 0]

    """
    if not isinstance(j, Sequence) or not all(isinstance(i, Integral) for i in j):
        raise TypeError("j must be a sequence of integers.")

    if any(i < 0 for i in j):
        raise ValueError("j must be non-negative integers.")

    n = [int((-1 + (1 + 8 * i) ** 0.5) / 2) for i in j]
    m = [2 * i - n_i * (n_i + 2) for n_i, i in zip(n, j)]

    return n, m