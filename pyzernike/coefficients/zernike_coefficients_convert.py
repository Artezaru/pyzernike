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

from typing import Sequence, Union

import numpy

from ..core.parsing import _as_coefficients, _as_indexing, _check_boolean
from ..core.core_coefficients import _core_convert_coefficients


def zernike_coefficients_convert(
    coefficients: Union[Sequence[float], numpy.ndarray],
    *,
    input_indexing: str = "OSA",
    output_indexing: str = "OSA",
    input_normalized: bool = False,
    output_normalized: bool = False,
) -> numpy.ndarray:
    r"""
    Convert a vector of coefficients of Zernike polynomials between indexing conventions (OSA/ANSI, Noll, Fringe)
    and between the normalized and the not normalized polynomials.

    The function :math:`f = \sum_j c_j Z_j` is unchanged by the conversion:

    - **Indexing.** The coefficient of each polynomial :math:`Z_{n}^{m}` is moved from its index in
      ``input_indexing`` to its index in ``output_indexing`` (see :func:`pyzernike.zernike_index_to_order`).
      The vectors start at the first index of each convention (0 for OSA/ANSI, 1 for Noll and Fringe).
    - **Normalization.** With the factors of Noll :math:`N_{n}^{m} = \sqrt{(2 - \delta_{m0})(n + 1)}`,
      :math:`c_j Z_j = c'_j (N_j Z_j)`, so :math:`c'_j = c_j / N_j` from the not normalized to the normalized
      polynomials, and :math:`c_j = c'_j N_j` in the other direction.

    **Length of the output.** The coefficients of the polynomials that are not in the input are set to zero.

    - If ``input_indexing`` and ``output_indexing`` are the same, the output has the length of the input.
    - If ``output_indexing`` is OSA/ANSI or Noll, the output contains all the polynomials up to the maximum radial
      order :math:`N` of the input polynomials, so its length is :math:`\frac{(N+1)(N+2)}{2}`. In the OSA/ANSI
      ordering, the output can be given directly to :func:`pyzernike.zernike_evaluate`.
    - If ``output_indexing`` is Fringe, the output ends at the largest Fringe index of the input polynomials (the
      Fringe ordering is not sorted by radial order: for example, :math:`Z_4^0` has the index 9 and :math:`Z_3^3`
      the index 10).

    .. seealso::

        - :func:`pyzernike.zernike_index_to_order` and :func:`pyzernike.zernike_order_to_index` for the indexing
          conventions.
        - :func:`pyzernike.zernike_normalization` for the normalization factors of Noll.
        - The page :doc:`../../mathematical_description/conventions` for the indexing conventions and the normalization.
        - The page :doc:`../../mathematical_description/coefficients` for the operations on the coefficients.

    Parameters
    ----------
    coefficients : Union[Sequence[float], numpy.ndarray]
        The coefficients in the ``input_indexing`` ordering, as a non-empty 1D array of finite real numbers.

    input_indexing : str, optional
        The indexing of the input coefficients: ``"OSA"`` (or ``"ANSI"``), ``"Noll"`` or ``"Fringe"``
        (case-insensitive). Default is ``"OSA"``.

    output_indexing : str, optional
        The indexing of the output coefficients: ``"OSA"`` (or ``"ANSI"``), ``"Noll"`` or ``"Fringe"``
        (case-insensitive). Default is ``"OSA"``.

    input_normalized : bool, optional
        If True, the input coefficients are those of the normalized polynomials. Default is False.

    output_normalized : bool, optional
        If True, the output coefficients are those of the normalized polynomials. Default is False.

    Returns
    -------
    numpy.ndarray
        The converted coefficients as a 1D array of ``numpy.float64``.

    Raises
    ------
    TypeError
        If ``coefficients`` can not be converted to an array of real numbers, if ``input_indexing`` or
        ``output_indexing`` is not a string, or if ``input_normalized`` or ``output_normalized`` is not a boolean.

    ValueError
        If ``coefficients`` is not a non-empty 1D array of finite values, or if ``input_indexing`` or
        ``output_indexing`` is not a known convention.

    Examples
    --------
    Convert the coefficients of the first 11 Noll polynomials (up to the spherical aberration :math:`Z_4^0`)
    to the OSA/ANSI ordering, to evaluate them:

    .. code-block:: python

        import numpy
        from pyzernike import zernike_coefficients_convert, zernike_evaluate

        noll = numpy.array([0.0, 0.1, -0.2, 0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.05])  # Noll indices 1 to 11
        osa = zernike_coefficients_convert(noll, input_indexing="Noll", output_indexing="OSA")  # 15 coefficients

    Convert normalized coefficients (e.g. given by a measurement instrument) to the not normalized polynomials:

    .. code-block:: python

        raw = zernike_coefficients_convert(osa, input_normalized=True, output_normalized=False)

    """
    coefficients = _as_coefficients(coefficients)
    input_indexing = _as_indexing(input_indexing)
    output_indexing = _as_indexing(output_indexing)
    input_normalized = _check_boolean(input_normalized, "input_normalized")
    output_normalized = _check_boolean(output_normalized, "output_normalized")

    return _core_convert_coefficients(
        coefficients=coefficients,
        input_indexing=input_indexing,
        output_indexing=output_indexing,
        input_normalized=input_normalized,
        output_normalized=output_normalized,
    )