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
from typing import List, Sequence, Tuple, Union, overload

from ..core.parsing import _as_indexing, _as_integers
from ..core.core_indexing import FIRST_INDEX, _core_index_to_order


@overload
def zernike_index_to_order(j: Integral, indexing: str = "OSA") -> Tuple[int, int]: ...
@overload
def zernike_index_to_order(j: Sequence[Integral], indexing: str = "OSA") -> Tuple[List[int], List[int]]: ...

def zernike_index_to_order(
    j: Union[Integral, Sequence[Integral]],
    indexing: str = "OSA",
) -> Union[Tuple[int, int], Tuple[List[int], List[int]]]:
    r"""
    Convert indices of a Zernike polynomial ordering to their corresponding Zernike orders :math:`(n, m)`.

    Three indexing conventions are available (see the page :doc:`../../mathematical_description/conventions` for the
    formulas and the details):

    - ``"OSA"`` (or ``"ANSI"``, default): the OSA/ANSI standard [1]_, starting at 0, sorted by increasing :math:`n`,
      then by increasing :math:`m`, with :math:`j = \frac{n(n + 2) + m}{2}`.
    - ``"Noll"``: the indices of Noll [2]_, starting at 1, sorted by increasing :math:`n`, then by increasing
      :math:`|m|`, with the even indices for :math:`m > 0` (cosine) and the odd indices for :math:`m < 0` (sine).
    - ``"Fringe"``: the Fringe (University of Arizona) indices, starting at 1, sorted by increasing
      :math:`\frac{n + |m|}{2}`, then by decreasing :math:`|m|`, with
      :math:`j = \left(1 + \frac{n + |m|}{2}\right)^2 - 2|m| + [m < 0]`. This ordering is not sorted by radial order.

    The first Zernike polynomials in each convention are:

    +-----+-----+-------+------+--------+
    |  n  |  m  |  OSA  | Noll | Fringe |
    +=====+=====+=======+======+========+
    |  0  |  0  |   0   |  1   |   1    |
    +-----+-----+-------+------+--------+
    |  1  | -1  |   1   |  3   |   3    |
    +-----+-----+-------+------+--------+
    |  1  |  1  |   2   |  2   |   2    |
    +-----+-----+-------+------+--------+
    |  2  | -2  |   3   |  5   |   6    |
    +-----+-----+-------+------+--------+
    |  2  |  0  |   4   |  4   |   4    |
    +-----+-----+-------+------+--------+
    |  2  |  2  |   5   |  6   |   5    |
    +-----+-----+-------+------+--------+
    |  3  | -3  |   6   |  9   |   11   |
    +-----+-----+-------+------+--------+
    |  3  | -1  |   7   |  7   |   8    |
    +-----+-----+-------+------+--------+
    |  3  |  1  |   8   |  8   |   7    |
    +-----+-----+-------+------+--------+
    |  3  |  3  |   9   |  10  |   10   |
    +-----+-----+-------+------+--------+
    |  4  |  0  |  12   |  11  |   9    |
    +-----+-----+-------+------+--------+

    The functions computing all the polynomials up to an order (e.g. :func:`pyzernike.zernike_polynomial_up_to_order`)
    and the functions working on vectors of coefficients always use the OSA/ANSI ordering. The vectors of coefficients
    are converted between the conventions with :func:`pyzernike.zernike_coefficients_convert`.

    The conversions use exact integer arithmetic (``math.isqrt``), so that the result is exact for any index.

    **Shape of the input and of the output.** ``j`` can be a single integer or a sequence of integers (list, tuple or
    1D numpy array):

    - If ``j`` is a single integer, a tuple ``(n, m)`` of integers is returned.
    - If ``j`` is a sequence, a tuple ``(n, m)`` of lists of integers is returned.

    .. seealso::

        - :func:`pyzernike.zernike_order_to_index` for the inverse conversion.
        - :func:`pyzernike.zernike_polynomials_count` for the number of polynomials up to an order.
        - :func:`pyzernike.zernike_coefficients_convert` to convert vectors of coefficients between the conventions.
        - The page :doc:`../../mathematical_description/conventions` for the indexing conventions.

    Parameters
    ----------
    j : Union[Integral, Sequence[Integral], numpy.ndarray]
        The index, or the sequence (list, tuple or 1D numpy array) of indices, of the Zernike polynomials. Must be
        greater than or equal to 0 (OSA/ANSI) or 1 (Noll and Fringe).

    indexing : str, optional
        The indexing convention: ``"OSA"`` (or ``"ANSI"``), ``"Noll"`` or ``"Fringe"``, case insensitive.
        Default is ``"OSA"``.

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
        If ``j`` is not an integer, a sequence of integers or a 1D numpy array of integers, or if ``indexing``
        is not a string.

    ValueError
        If an index in ``j`` is lower than the first index of the convention, or if ``indexing`` is unknown.

    References
    ----------
    .. [1] L. N. Thibos, R. A. Applegate, J. T. Schwiegerling and R. Webb, "Standards for reporting the optical
       aberrations of eyes," Journal of Refractive Surgery 18(5), S652-S660 (2002).
       https://doi.org/10.3928/1081-597X-20020901-30

    .. [2] R. J. Noll, "Zernike polynomials and atmospheric turbulence," Journal of the Optical Society
       of America 66(3), 207-211 (1976). https://doi.org/10.1364/JOSA.66.000207

    Examples
    --------
    Convert a single index:

    .. code-block:: python

        from pyzernike import zernike_index_to_order

        n, m = zernike_index_to_order(4)
        print(n, m)  # Output: 2 0

    Convert several indices:

    .. code-block:: python

        j = [2, 3, 4]
        n, m = zernike_index_to_order(j)
        print(n)  # Output: [1, 2, 2]
        print(m)  # Output: [1, -2, 0]

    Convert Noll indices:

    .. code-block:: python

        n, m = zernike_index_to_order([1, 2, 3, 4, 11], indexing="Noll")
        print(n)  # Output: [0, 1, 1, 2, 4]
        print(m)  # Output: [0, 1, -1, 0, 0]

    """
    indexing = _as_indexing(indexing)
    first = FIRST_INDEX[indexing]
    indices = _as_integers(j, "j")

    # Single integer
    if isinstance(indices, int):
        if indices < first:
            raise ValueError(f"j must be an integer greater than or equal to {first} for the {indexing} indexing.")
        n, m = _core_index_to_order(j=[indices], indexing=indexing)
        return n[0], m[0]

    # Sequence of integers
    if any(i < first for i in indices):
        raise ValueError(f"j must contain integers greater than or equal to {first} for the {indexing} indexing.")
    return _core_index_to_order(j=indices, indexing=indexing)