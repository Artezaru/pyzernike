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

def core_polynomial(
        rho: numpy.ndarray,
        theta: numpy.ndarray,
        n: Sequence[int], 
        m: Sequence[int], 
        rho_derivative: Sequence[int], 
        theta_derivative: Sequence[int],
        flag_radial: bool,
    ) -> List[numpy.ndarray]:
    r"""

    .. warning::

        This method is a core function of ``pyzernike`` that computes Zernike polynomials for given `n`, `m`, `rho_derivative`, and `theta_derivative` values.
        It is not a method designed to be use by the users directly, but rather a helper function for the optimized polynomial computation.

        Please ensure that you understand the mathematical background of Zernike polynomials before using this function.
        ``rho`` and ``theta`` are expected to be numpy arrays with float64 dtype into the valid domain :math:`0 \leq \rho \leq 1`, with finite values, not NaN values and with the same shape.
        ``n``, ``m``, ``rho_derivative`` and ``theta_derivative`` are expected to be sequences of integers with the same length and valid values.

    .. seealso::

        - :func:`pyzernike.radial_polynomial` for the radial Zernike polynomial computation.
        - :func:`pyzernike.zernike_polynomial` for the full Zernike polynomial computation.
        - :func:`pyzernike.xy_zernike_polynomial` for the Zernike polynomial computation in Cartesian coordinates.

    Assemble the Zernike polynomial :math:`Z_{n}^{m}(\rho, \theta)` or the radial Zernike polynomial :math:`R_{n}^{m}(\rho)` if the flag `flag_radial` is set to True.

    The Zernike polynomial is defined as follows:

    .. math::

        Z_{n}^{m}(\rho, \theta) = R_{n}^{m}(\rho) \cos(m \theta) \quad \text{if} \quad m \geq 0
    
    .. math::

        Z_{n}^{m}(\rho, \theta) = R_{n}^{-m}(\rho) \sin(-m \theta) \quad \text{if} \quad m < 0

    The derivative of order (derivative (a)) of the Zernike polynomial with respect to rho and order (derivative (b)) with respect to theta is defined as follows :

    .. math::

        \frac{\partial^{a}\partial^{b}Z_{n}^{m}(\rho, \theta)}{\partial \rho^{a} \partial \theta^{b}} = \frac{\partial^{a}R_{n}^{m}(\rho)}{\partial \rho^{a}} \frac{\partial^{b}\cos(m \theta)}{\partial \theta^{b}} \quad \text{if} \quad m > 0

    If :math:`|m| > n` or :math:`n < 0`, or :math:`(n - m)` is odd, the output is a zeros array with the given shape.

    The radial Zernike polynomial is defined as follows:

    .. math::

        R_{n}^{m}(\rho) = \sum_{k=0}^{(n-m)/2} \frac{(-1)^k (n-k)!}{k! ((n+m)/2 - k)! ((n-m)/2 - k)!} \rho^{n-2k}

    The derivative of order (derivative (a)) of the radial Zernike polynomial is defined as follows :

    .. math::

        \frac{d^{a}R_{n}^{m}(\rho)}{d\rho^{a}} = \sum_{k=0}^{(n-m)/2} \frac{(-1)^k (n-k)!}{k! ((n+m)/2 - k)! ((n-m)/2 - k)!} (n-2k) (n-2k-1) \ldots (n-2k-a+1) \rho^{n-2k-a}

    The computation of the factorial is done using the function :func:`scipy.special.gammaln` for better performance and stability.

    .. math::

        \text{log}(n!) = \text{gammaln}(n+1)

    So the coefficient of the radial polynomial is computed as follows:

    .. math::

        \frac{(n-k)!}{k! ((n+m)/2 - k)! ((n-m)/2 - k)!} = \text{exp} (\text{gammaln}(n-k+1) - \text{gammaln}(k+1) - \text{gammaln}((n+m)/2 - k + 1) - \text{gammaln}((n-m)/2 - k + 1))

    .. note::

        The fonction is designed to precompute the useful terms for the Zernike polynomials, such as the powers of rho, the cosine and sine terms, and the logarithm of the factorials.

    Parameters
    ----------
    rho : numpy.ndarray (N-D array)
        The radial coordinate values with shape (...,) and float64 dtype.

    theta : numpy.ndarray (N-D array)
        The angular coordinate values with shape (...,) and float64 dtype. ONLY USED IF `flag_radial` IS False.
    
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

    Returns
    -------
    List[numpy.ndarray]
        A list of numpy.ndarray containing the Zernike polynomials for each (n, m, rho_derivative, theta_derivative) tuple, or the radial Zernike polynomials if `flag_radial` is True.
        Each polynomial has the shape specified by `shape`.
    """
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

    # Construct the sets for the useful terms
    powers_exponents = set() 
    cosine_frequencies = set()
    sine_frequencies = set()
    factorials_integers = set()
    max_n = None
    max_m = None

    for idx in range(len(n)):
        # Extract the current indices
        n_idx = n[idx]
        m_idx = m[idx]
        dr_idx = rho_derivative[idx]

        # Exponents and factorials sets
        dr_idx = rho_derivative[idx]
        if n_idx >= dr_idx:
            max_k = min((n_idx - abs(m_idx)) // 2, (n_idx - dr_idx) // 2)
            powers_exponents.update([n_idx - 2 * k - dr_idx for k in range(max_k + 1)])
            for k in range(max_k + 1):
                factorials_integers.update([n_idx - k, k, (n_idx + abs(m_idx)) // 2 - k, (n_idx - abs(m_idx)) // 2 - k, n_idx - 2 * k, n_idx - 2 * k - dr_idx])

        # Cosine frequency and sine frequency sets
        if not flag_radial:
            # Extract the angular theta derivative
            dt_idx = theta_derivative[idx]

            # Add frequencies in the cosine and sine terms sets
            if (m_idx > 0 and dt_idx % 2 == 0) or (m_idx < 0 and dt_idx % 2 == 1):
                cosine_frequencies.add(abs(m_idx))
            elif (m_idx < 0 and dt_idx % 2 == 0) or (m_idx > 0 and dt_idx % 2 == 1):
                sine_frequencies.add(abs(m_idx))

        # Updating the maximum values for n and m
        if max_n is None or n_idx > max_n:
            max_n = n_idx
        if max_m is None or abs(m_idx) > max_m:
            max_m = abs(m_idx)

    # Construct the precomputed terms
    rho_powers_indices_map = numpy.zeros(max_n + 1, dtype=int)
    for index, exponent in enumerate(powers_exponents):
        rho_powers_indices_map[exponent] = index
    rho_powers = numpy.power(rho[..., numpy.newaxis], list(powers_exponents))

    log_factorials_indices_map = numpy.zeros((max_n + 1,), dtype=int)
    for index, integer in enumerate(factorials_integers):
        log_factorials_indices_map[integer] = index
    log_factorials = gammaln(numpy.array(list(factorials_integers), dtype=float) + 1)

    # If flag_radial is True, we do not compute the angular terms
    if not flag_radial:
            cosine_terms_indices_map = numpy.zeros((max_m + 1,), dtype=int)
            for index, frequency in enumerate(cosine_frequencies):
                cosine_terms_indices_map[frequency] = index
            cosine_terms = numpy.cos(list(cosine_frequencies) * theta[..., numpy.newaxis])

            sine_terms_indices_map = numpy.zeros((max_m + 1,), dtype=int)
            for index, frequency in enumerate(sine_frequencies):    
                sine_terms_indices_map[frequency] = index
            sine_terms = numpy.sin(list(sine_frequencies) * theta[..., numpy.newaxis])

    # =================================================================
    # Boucle over the polynomials to compute
    # =================================================================
    for idx in range(len(n)):
        # Extract the n, m, rho_derivative
        n_idx = n[idx]
        m_idx = m[idx]
        rho_derivative_idx = rho_derivative[idx]

        # Case of n < 0, (n - m) is odd or |m| > n
        if n_idx < 0 or (n_idx - m_idx) % 2 != 0 or abs(m_idx) > n_idx:
            output.append(numpy.zeros(rho.shape, dtype=numpy.float64))
            continue

        # Case for derivatives of order greater than n_idx
        if rho_derivative_idx > n_idx:
            output.append(numpy.zeros(rho.shape, dtype=numpy.float64))
            continue

        # Case for radial polynomial only
        if flag_radial and m_idx < 0:
            output.append(numpy.zeros(rho.shape, dtype=numpy.float64))
            continue

        # Compute the number of terms
        s = min((n_idx - abs(m_idx)) // 2, (n_idx - rho_derivative_idx) // 2) # No computation for terms derivated more than the index of the polynomial
        k = numpy.arange(0, s + 1)

        # Compute the coefficients
        log_k_coef = log_factorials[log_factorials_indices_map[n_idx - k]] - \
                     log_factorials[log_factorials_indices_map[k]] - \
                     log_factorials[log_factorials_indices_map[(n_idx + abs(m_idx)) // 2 - k]] - \
                     log_factorials[log_factorials_indices_map[(n_idx - abs(m_idx)) // 2 - k]]

        sign = 1 - 2 * (k % 2)

        if rho_derivative_idx != 0:
            log_k_coef += log_factorials[log_factorials_indices_map[n_idx - 2 * k]] - log_factorials[log_factorials_indices_map[n_idx - 2 * k - rho_derivative_idx]]

        coef = sign * numpy.exp(log_k_coef)

        # Compute the rho power
        exponent = n_idx - 2 * k - rho_derivative_idx
        rho_orders = rho_powers[..., rho_powers_indices_map[exponent]]

        # Assemble the radial polynomial if flag_radial is True
        result = numpy.tensordot(rho_orders, coef, axes=[[-1], [0]])

        # Theta part of the Zernike polynomial
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
                    cosine = cosine_terms[..., cosine_terms_indices_map[abs(m_idx)]]
                elif theta_derivative_idx % 4 == 0:
                    cosine = (abs(m_idx) ** theta_derivative_idx) * cosine_terms[..., cosine_terms_indices_map[abs(m_idx)]]
                elif theta_derivative_idx % 4 == 1:
                    cosine = - (abs(m_idx) ** theta_derivative_idx) * sine_terms[..., sine_terms_indices_map[abs(m_idx)]]
                elif theta_derivative_idx % 4 == 2:
                    cosine = - (abs(m_idx) ** theta_derivative_idx) * cosine_terms[..., cosine_terms_indices_map[abs(m_idx)]]
                else:
                    cosine = (abs(m_idx) ** theta_derivative_idx) * sine_terms[..., sine_terms_indices_map[abs(m_idx)]]
                
            if m_idx < 0:
                if theta_derivative_idx == 0:
                    cosine = sine_terms[..., sine_terms_indices_map[abs(m_idx)]]
                elif theta_derivative_idx % 4 == 0:
                    cosine = (abs(m_idx) ** theta_derivative_idx) * sine_terms[..., sine_terms_indices_map[abs(m_idx)]]
                elif theta_derivative_idx % 4 == 1:
                    cosine = (abs(m_idx) ** theta_derivative_idx) * cosine_terms[..., cosine_terms_indices_map[abs(m_idx)]]
                elif theta_derivative_idx % 4 == 2:
                    cosine = - (abs(m_idx) ** theta_derivative_idx) * sine_terms[..., sine_terms_indices_map[abs(m_idx)]]
                else:
                    cosine = - (abs(m_idx) ** theta_derivative_idx) * cosine_terms[..., cosine_terms_indices_map[abs(m_idx)]]

            # Multiply the radial polynomial by the cosine factor
            result *= cosine
        
        # Compute the polynomial
        output.append(result)

    return output


