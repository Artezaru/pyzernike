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

import numpy
from typing import Sequence, List
from scipy.special import gammaln

from .core_create_precomputing_sets import core_create_precomputing_sets

def core_polynomial(
        rho: numpy.ndarray,
        theta: numpy.ndarray,
        n: Sequence[int], 
        m: Sequence[int], 
        rho_derivative: Sequence[int], 
        theta_derivative: Sequence[int],
        flag_radial: bool,
        precompute: bool,
    ) -> List[numpy.ndarray]:
    r"""

    Assemble the Zernike polynomial :math:`Z_{n}^{m}(\rho, \theta)` or the radial Zernike polynomial :math:`R_{n}^{m}(\rho)` (if the flag `flag_radial` 
    is set to True) for each given tuple of (n, m, rho_derivative, theta_derivative) in the input lists.

    .. warning::

        This method is a core function of ``pyzernike`` that is not designed to be use by the users directly. 
        No test is done on the input parameters. Please use the high level functions.

    .. seealso::

        - :func:`pyzernike.radial_polynomial` for the radial Zernike polynomial computation.
        - :func:`pyzernike.zernike_polynomial` for the full Zernike polynomial computation.
        - The page :doc:`../../mathematical_description` in the documentation for the mathematical description of the Zernike polynomials.

    - ``rho`` and ``theta`` are expected to be floating point type numpy arrays of the same shape, same dtype and in the range [0, 1] for ``rho``.
    - ``n``, ``m``, ``rho_derivative`` and ``theta_derivative`` are expected to be sequences of integers of the same length and valid values.

    The function is designed to precompute the useful terms for the Zernike polynomials, such as the powers of rho, the cosine and sine terms, and the logarithm of the factorials.

    Parameters
    ----------
    rho : numpy.ndarray (N-D array)
        The radial coordinate values with shape (...,) and floating point dtype.

    theta : numpy.ndarray (N-D array)
        The angular coordinate values with shape (...,) and floating point dtype. ONLY USED IF `flag_radial` IS False.

    n : Sequence[int]
        A list of integers containing the order `n` of each radial Zernike polynomial to compute.

    m : Sequence[int]
        A list of integers containing the degree `m` of each radial Zernike polynomial to compute.

    rho_derivative : Sequence[int]
        A list of integers containing the order of the radial derivative to compute for each radial Zernike polynomial.
        If `rho_derivative` is None, no radial derivative is computed.

    theta_derivative : Sequence[int]
        A list of integers containing the order of the angular derivative to compute for each Zernike polynomial.
        If `theta_derivative` is None, no angular derivative is computed. ONLY USED IF `flag_radial` IS False.

    flag_radial : bool
        If True, the radial Zernike polynomial :math:`R_{n}^{m}(\rho)` is computed instead of the full Zernike polynomial :math:`Z_{n}^{m}(\rho, \theta)`.
        If False, the full Zernike polynomial :math:`Z_{n}^{m}(\rho, \theta)` is computed, which includes the angular part with the cosine and sine terms.

    precompute : bool
        If True, the useful terms for the Zernike polynomials are precomputed to optimize the computation.
        This is useful when computing multiple Zernike polynomials with the same `rho` and `theta` values.
        If False, the useful terms are computed on-the-fly for each polynomial, which may be slower but avoid memory overhead.

    Returns
    -------
    List[numpy.ndarray]
        A list of numpy.ndarray containing the Zernike polynomials for each (n, m, rho_derivative, theta_derivative) tuple, or the radial Zernike polynomials if `flag_radial` is True.
        Each polynomial has the shape of `rho` (and `theta` if `flag_radial` is False).

    """
    # Extract the shape and dtype of the input arrays
    shape = rho.shape
    dtype = rho.dtype

    # Create the output list
    output = []

    # =================================================================
    # Precomputation of the useful terms
    # =================================================================
    #
    # rho_powers_indices_map : numpy.ndarray (1-D array)
    #     An array of shape=(max(n) + 1,) containing the indices of the rho powers in `rho_powers` for a given exponent.
    #     This is used to map the computed radial polynomial to the precomputed rho powers.
    # 
    # rho_powers : numpy.ndarray (2-D array)
    #     An array of shape=(..., Nexponents) containing the precomputed powers of rho for the useful exponents.
    # 
    # cosine_terms_indices_map : numpy.ndarray (1-D array)
    #     An array of shape=(max(m) + 1,) containing the indices of the cosine terms in `cosine_terms` for a given degree.
    #     This is used to map the computed angular polynomial coefficients to the precomputed cosine terms. ONLY USED IF `flag_radial` IS False.
    # 
    # cosine_terms : numpy.ndarray (2-D array)
    #     An array of shape=(..., Ncosine_terms) containing the cosine terms for the useful angular polynomials. ONLY USED IF `flag_radial` IS False.
    #   
    # sine_terms_indices_map : numpy.ndarray (1-D array)
    #     An array of shape=(max(m) + 1,) containing the indices of the sine terms in `sine_terms` for a given degree.
    #     This is used to map the computed angular polynomial coefficients to the precomputed sine terms. ONLY USED IF `flag_radial` IS False.
    # 
    # sine_terms : numpy.ndarray (2-D array)
    #     An array of shape=(..., Nsine_terms) containing the sine terms for the useful angular polynomials. ONLY USED IF `flag_radial` IS False.
    # 
    # log_factorials_indices_map : numpy.ndarray (1-D array)
    #     An array of shape=(max(n) + 1,) containing the indices of the logarithm of the factorials in `log_factorials` for a given integer.
    #     This is used to map the computed radial polynomial coefficients to the precomputed logarithm of the factorials.
    # 
    # log_factorials : numpy.ndarray (1-D array)
    #     An array of shape=(Nfactorials,) containing the logarithm of the factorials for the useful integers.
    # 
    # =================================================================

    if precompute:

        # Construct the sets for the useful terms
        precomputed_sets = core_create_precomputing_sets(n, m, rho_derivative, theta_derivative, flag_radial)
        powers_exponents, cosine_frequencies, sine_frequencies, factorials_integers, max_n, max_m = precomputed_sets

        # Precompute the rho powers
        rho_powers_indices_map = numpy.zeros(max_n + 1, dtype=int)
        for index, exponent in enumerate(powers_exponents):
            rho_powers_indices_map[exponent] = index
        rho_powers = numpy.power(rho[..., numpy.newaxis], list(powers_exponents))

        # Precompute the logarithm of the factorials
        log_factorials_indices_map = numpy.zeros((max_n + 1,), dtype=int)
        for index, integer in enumerate(factorials_integers):
            log_factorials_indices_map[integer] = index
        log_factorials = gammaln(numpy.array(list(factorials_integers), dtype=dtype) + 1)

        # If flag_radial is True, we do not compute the angular terms
        if not flag_radial:

            # Precompute the cosine terms
            cosine_terms_indices_map = numpy.zeros((max_m + 1,), dtype=int)
            for index, frequency in enumerate(cosine_frequencies):
                cosine_terms_indices_map[frequency] = index
            cosine_terms = numpy.cos(list(cosine_frequencies) * theta[..., numpy.newaxis])

            # Precompute the sine terms
            sine_terms_indices_map = numpy.zeros((max_m + 1,), dtype=int)
            for index, frequency in enumerate(sine_frequencies):    
                sine_terms_indices_map[frequency] = index
            sine_terms = numpy.sin(list(sine_frequencies) * theta[..., numpy.newaxis])


    # =================================================================
    # Boucle over the polynomials to compute
    # =================================================================
    # Loop over the input lists to compute each polynomial
    for idx in range(len(n)):

        # Extract the n, m, rho_derivative
        n_idx = n[idx]
        m_idx = m[idx]
        rho_derivative_idx = rho_derivative[idx]

        # Case of n < 0, (n - m) is odd or |m| > n
        if n_idx < 0 or (n_idx - m_idx) % 2 != 0 or abs(m_idx) > n_idx:
            output.append(numpy.zeros(shape, dtype=dtype))
            continue

        # Case for derivatives of order greater than n_idx
        if rho_derivative_idx > n_idx:
            output.append(numpy.zeros(shape, dtype=dtype))
            continue

        # Case for radial polynomial and negative m
        if flag_radial and m_idx < 0:
            output.append(numpy.zeros(shape, dtype=dtype))
            continue

        # Compute the number of terms of the radial polynomial
        s = min((n_idx - abs(m_idx)) // 2, (n_idx - rho_derivative_idx) // 2) # No computation for terms derivated more than the index of the polynomial
        k = numpy.arange(0, s + 1)

        # Compute the coefficients of the radial polynomial
        if precompute:
            log_k_coef = log_factorials[log_factorials_indices_map[n_idx - k]] - \
                         log_factorials[log_factorials_indices_map[k]] - \
                         log_factorials[log_factorials_indices_map[(n_idx + abs(m_idx)) // 2 - k]] - \
                         log_factorials[log_factorials_indices_map[(n_idx - abs(m_idx)) // 2 - k]]
        else:
            log_k_coef = gammaln(n_idx - k + 1) - \
                         gammaln(k + 1) - \
                         gammaln((n_idx + abs(m_idx)) // 2 - k + 1) - \
                         gammaln((n_idx - abs(m_idx)) // 2 - k + 1)

        sign = 1 - 2 * (k % 2)

        if rho_derivative_idx != 0:
            if precompute:
                log_k_coef += log_factorials[log_factorials_indices_map[n_idx - 2 * k]] - \
                              log_factorials[log_factorials_indices_map[n_idx - 2 * k - rho_derivative_idx]]
            else:
                log_k_coef += gammaln(n_idx - 2 * k + 1) - \
                              gammaln(n_idx - 2 * k - rho_derivative_idx + 1)

        coef = sign * numpy.exp(log_k_coef)

        # Compute the rho power terms
        exponent = n_idx - 2 * k - rho_derivative_idx
        if precompute:
            rho_orders = rho_powers[..., rho_powers_indices_map[exponent]]
        else:
            rho_orders = numpy.power(rho[..., numpy.newaxis], list(exponent))

        # Assemble the radial polynomial
        result = numpy.tensordot(rho_orders, coef, axes=[[-1], [0]])

        # Theta part of the Zernike polynomial if flag_radial is False
        if not flag_radial:

            # Extract the angular theta derivative
            theta_derivative_idx = theta_derivative[idx]    
            
            # According to the angular derivative, we compute the cosine factor
            if m_idx == 0:
                if theta_derivative_idx == 0:
                    cosine = 1.0
                else:
                    cosine = 0.0
                
            if m_idx > 0:
                if theta_derivative_idx == 0:
                    cosine = cosine_terms[..., cosine_terms_indices_map[abs(m_idx)]] if precompute else numpy.cos(m_idx * theta)
                elif theta_derivative_idx % 4 == 0:
                    if precompute:
                        cosine = (abs(m_idx) ** theta_derivative_idx) * cosine_terms[..., cosine_terms_indices_map[abs(m_idx)]] 
                    else:
                        cosine = (abs(m_idx) ** theta_derivative_idx) * numpy.cos(m_idx * theta)
                elif theta_derivative_idx % 4 == 1:
                    if precompute:
                        cosine = - (abs(m_idx) ** theta_derivative_idx) * sine_terms[..., sine_terms_indices_map[abs(m_idx)]]
                    else:
                        cosine = - (abs(m_idx) ** theta_derivative_idx) * numpy.sin(m_idx * theta)
                elif theta_derivative_idx % 4 == 2:
                    if precompute:
                        cosine = - (abs(m_idx) ** theta_derivative_idx) * cosine_terms[..., cosine_terms_indices_map[abs(m_idx)]]
                    else:
                        cosine = - (abs(m_idx) ** theta_derivative_idx) * numpy.cos(m_idx * theta)
                else:
                    if precompute:
                        cosine = (abs(m_idx) ** theta_derivative_idx) * sine_terms[..., sine_terms_indices_map[abs(m_idx)]]
                    else:
                        cosine = (abs(m_idx) ** theta_derivative_idx) * numpy.sin(m_idx * theta)

            if m_idx < 0:
                if theta_derivative_idx == 0:
                    cosine = sine_terms[..., sine_terms_indices_map[abs(m_idx)]] if precompute else numpy.sin(abs(m_idx) * theta)
                elif theta_derivative_idx % 4 == 0:
                    if precompute:
                        cosine = (abs(m_idx) ** theta_derivative_idx) * sine_terms[..., sine_terms_indices_map[abs(m_idx)]]
                    else:
                        cosine = (abs(m_idx) ** theta_derivative_idx) * numpy.sin(abs(m_idx) * theta)
                elif theta_derivative_idx % 4 == 1:
                    if precompute:
                        cosine = (abs(m_idx) ** theta_derivative_idx) * cosine_terms[..., cosine_terms_indices_map[abs(m_idx)]]
                    else:
                        cosine = (abs(m_idx) ** theta_derivative_idx) * numpy.cos(abs(m_idx) * theta)
                elif theta_derivative_idx % 4 == 2:
                    if precompute:
                        cosine = - (abs(m_idx) ** theta_derivative_idx) * sine_terms[..., sine_terms_indices_map[abs(m_idx)]]
                    else:
                        cosine = - (abs(m_idx) ** theta_derivative_idx) * numpy.sin(abs(m_idx) * theta)
                else:
                    if precompute:
                        cosine = - (abs(m_idx) ** theta_derivative_idx) * cosine_terms[..., cosine_terms_indices_map[abs(m_idx)]]
                    else:
                        cosine = - (abs(m_idx) ** theta_derivative_idx) * numpy.cos(abs(m_idx) * theta)

            # Multiply the radial polynomial by the cosine factor
            result *= cosine
        
        # Save the polynomial
        output.append(result)

    return output


