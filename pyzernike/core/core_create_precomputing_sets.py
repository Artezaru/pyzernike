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

from typing import Sequence, Set, Tuple

def core_create_precomputing_sets(
    n: Sequence[int],
    m: Sequence[int],
    rho_derivative: Sequence[int],
    theta_derivative: Sequence[int],
    flag_radial: bool = False,
) -> Tuple[Set[int], Set[int], Set[int], Set[int], int, int]:
    r"""
    Create the sets of usefull exponents, frequencies and integers for the computation of Zernike polynomials and their derivatives.

    .. warning::

        This method is a core function of ``pyzernike`` that is not designed to be use by the users directly.
        No test is done on the input parameters. Please use the high level functions.

    .. seealso::

        :func:`pyzernike.core.core_polynomial` for computing Zernike polynomials.

    For one defined Zernike polynomial of order ``n``, degree ``m`` and derivative with respect to rho ``a``, the usefull rho exponents are :

    .. math::

        \{ n - 2k - a \mid k = 0, 1, \ldots, \frac{n - |m|}{2} \}

    The useful integers for the factorials are :

    .. math::

        \{ n - k, k, \frac{n + |m|}{2} - k, \frac{n - |m|}{2} - k, n - 2k, n - 2k - a \mid k = 0, 1, \ldots, \frac{n - |m|}{2} \}

    if :math:`n \geq a` and :math:`n \geq |m|` and :math:`(n - m)` is even, otherwise the output is a zeros array with the same shape as :math:`\rho`.

    For the angular part, the usefull frequencies for the cosine and sine terms are :math:`|m|` depending on the parity of ``theta_derivative`` and the sign of ``m``.

    Parameters
    ----------
    n : Sequence[int]
        The orders of the Zernike polynomials.

    m : Sequence[int]
        The degrees of the Zernike polynomials.

    rho_derivative : Sequence[int]
        The orders of the derivatives with respect to rho.

    theta_derivative : Sequence[int]
        The orders of the derivatives with respect to theta.

    flag_radial : bool, optional
        If True, computes the sets for radial polynomials only (no angular part). The output sine and cosine frequency sets will be empty.
        If False, computes the sets for full Zernike polynomials (including angular part).
        The default is False.

    Returns
    -------
    (Set[int], Set[int], Set[int], Set[int], int, int)
        A tuple containing:

        - ``powers_exponents_set``: A set of unique exponents for the powers of rho needed for the computations.
        - ``cosine_frequencies_set``: A set of unique frequencies for the cosine terms needed for the computations.
        - ``sine_frequencies_set``: A set of unique frequencies for the sine terms needed for the computations.
        - ``factorials_integers_set``: A set of unique integers for the factorials needed for the computations.
        - ``max_n``: The maximum order in ``n``.
        - ``max_m``: The maximum absolute degree in ``m``.

    """
    # Construct the sets for the useful terms
    powers_exponents: Set[int] = set() 
    cosine_frequencies: Set[int] = set()
    sine_frequencies: Set[int] = set()
    factorials_integers: Set[int] = set()
    max_n: int = -1
    max_m: int = 1

    for idx in range(len(n)):
        # Extract the n, m, dr values
        n_idx, m_idx, dr_idx = n[idx], m[idx], rho_derivative[idx]

        # Exponents and factorials sets
        if n_idx >= dr_idx:
            max_k = min((n_idx - abs(m_idx)) // 2, (n_idx - dr_idx) // 2)
            powers_exponents.update([n_idx - 2 * k - dr_idx for k in range(max_k + 1)])
            for k in range(max_k + 1):
                factorials_integers.update([
                        n_idx - k, k, 
                        (n_idx + abs(m_idx)) // 2 - k, 
                        (n_idx - abs(m_idx)) // 2 - k, 
                        n_idx - 2 * k,
                        n_idx - 2 * k - dr_idx
                    ])

        # Cosine frequency and sine frequency sets
        if not flag_radial:
            # Extract the dt value
            dt_idx = theta_derivative[idx]

            # Add frequencies in the cosine and sine terms sets
            if (m_idx > 0 and dt_idx % 2 == 0) or (m_idx < 0 and dt_idx % 2 == 1):
                cosine_frequencies.add(abs(m_idx))
            elif (m_idx < 0 and dt_idx % 2 == 0) or (m_idx > 0 and dt_idx % 2 == 1):
                sine_frequencies.add(abs(m_idx))

        # Updating the maximum values for n and m
        if n_idx > max_n:
            max_n = n_idx
        if abs(m_idx) > max_m:
            max_m = abs(m_idx)

    return powers_exponents, cosine_frequencies, sine_frequencies, factorials_integers, max_n, max_m
        