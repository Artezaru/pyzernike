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

from ..core.parsing import _as_indexing, _as_integers
from ..core.core_indexing import _core_order_to_index


@overload
def zernike_order_to_index(n: Integral, m: Integral, indexing: str = "OSA") -> int: ...
@overload
def zernike_order_to_index(n: Sequence[Integral], m: Sequence[Integral], indexing: str = "OSA") -> List[int]: ...

def zernike_order_to_index(
    n: Union[Integral, Sequence[Integral]],
    m: Union[Integral, Sequence[Integral]],
    indexing: str = "OSA",
) -> Union[int, List[int]]:
    r"""
    Convert Zernike orders :math:`(n, m)` to their corresponding indices in a Zernike polynomial ordering.

    Three indexing conventions are available (see the page :doc:`../../mathematical_description/conventions` for the
    details, and :func:`pyzernike.zernike_index_to_order` for the table of the first indices):

    - ``"OSA"`` (or ``"ANSI"``, default): the OSA/ANSI standard [1]_, starting at 0, with :math:`j = \frac{n(n + 2) + m}{2}`.
    - ``"Noll"``: the indices of Noll [2]_, starting at 1, with :math:`j = \frac{n(n + 1)}{2} + |m| + s`, where
      :math:`s \in \{0, 1\}` is chosen so that the even indices correspond to :math:`m > 0` (cosine) and the odd
      indices to :math:`m < 0` (sine).
    - ``"Fringe"``: the Fringe (University of Arizona) indices, starting at 1, with
      :math:`j = \left(1 + \frac{n + |m|}{2}\right)^2 - 2|m| + [m < 0]`.

    If :math:`n < 0`, :math:`|m| > n` or :math:`(n - m)` is odd, an error is raised.

    **Shape of the inputs and of the output.** ``n`` and ``m`` can be two integers, or two sequences of integers
    (lists, tuples or 1D numpy arrays) with the same length:

    - If ``n`` and ``m`` are integers, a single integer index is returned.
    - If ``n`` and ``m`` are sequences, a list of integer indices is returned.
    - Mixing an integer and a sequence is not allowed.

    .. seealso::

        - :func:`pyzernike.zernike_index_to_order` for the inverse conversion and the table of the first indices.
        - :func:`pyzernike.zernike_coefficients_convert` to convert vectors of coefficients between the conventions.
        - The page :doc:`../../mathematical_description/conventions` for the indexing conventions.

    Parameters
    ----------
    n : Union[Integral, Sequence[Integral]]
        The radial order, or the sequence of radial orders, of the Zernike polynomials.

    m : Union[Integral, Sequence[Integral]]
        The azimuthal frequency, or the sequence of azimuthal frequencies, of the Zernike polynomials.
        Must be of the same kind as ``n`` (integer or sequence) and, for sequences, of the same length.

    indexing : str, optional
        The indexing convention: ``"OSA"`` (or ``"ANSI"``), ``"Noll"`` or ``"Fringe"``, case insensitive.
        Default is ``"OSA"``.

    Returns
    -------
    Union[int, List[int]]
        If ``n`` and ``m`` are integers, the index ``j`` of the Zernike polynomial in the given convention.

        If ``n`` and ``m`` are sequences, the list of the indices, in the same order as ``n`` and ``m``.

    Raises
    ------
    TypeError
        If ``n`` or ``m`` is not an integer, a sequence of integers or a 1D numpy array of integers, if an
        integer is mixed with a sequence, or if ``indexing`` is not a string.

    ValueError
        If ``n`` and ``m`` do not have the same length, if a pair ``(n, m)`` is not a valid
        Zernike order (:math:`n < 0`, :math:`|m| > n` or :math:`(n - m)` odd), or if ``indexing`` is unknown.

    References
    ----------
    .. [1] L. N. Thibos, R. A. Applegate, J. T. Schwiegerling and R. Webb, "Standards for reporting the optical
       aberrations of eyes," Journal of Refractive Surgery 18(5), S652-S660 (2002).
       https://doi.org/10.3928/1081-597X-20020901-30

    .. [2] R. J. Noll, "Zernike polynomials and atmospheric turbulence," Journal of the Optical Society
       of America 66(3), 207-211 (1976). https://doi.org/10.1364/JOSA.66.000207

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

    Convert to Noll and Fringe indices:

    .. code-block:: python

        print(zernike_order_to_index([2, 2, 4], [-2, 2, 0], indexing="Noll"))  # Output: [5, 6, 11]
        print(zernike_order_to_index([2, 2, 4], [-2, 2, 0], indexing="Fringe"))  # Output: [6, 5, 9]

    """
    indexing = _as_indexing(indexing)
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

    indices = _core_order_to_index(n=n_list, m=m_list, indexing=indexing)
    return indices[0] if scalar_n else indices