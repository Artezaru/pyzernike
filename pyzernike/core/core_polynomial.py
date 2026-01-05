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
from typing import List, Optional, Tuple
from scipy.special import gammaln

from .core_create_precomputing_terms import core_create_precomputing_terms
from .core_corresponding_signed_integer_type import core_corresponding_signed_integer_type

def core_polynomial(
        rho: numpy.ndarray,
        theta: Optional[numpy.ndarray],
        n: numpy.array,
        m: numpy.array,
        rho_derivative: numpy.array,
        theta_derivative: Optional[numpy.array],
        flag_radial: bool,
        precompute: bool,
        float_type: type[numpy.floating]
    ) -> List[numpy.ndarray]:
    r"""

    Assemble the Zernike polynomial :math:`Z_{n}^{m}(\rho, \theta)` or the radial Zernike polynomial :math:`R_{n}^{m}(\rho)` (if the flag `flag_radial` 
    is set to True) for each given tuple of (n, m, rho_derivative, theta_derivative) in the input lists.

    .. warning::

        This method is a core function of ``pyzernike`` that is not designed to be use by the users directly. 
        Please use the high level functions.

    .. seealso::

        - :func:`pyzernike.radial_polynomial` for the radial Zernike polynomial computation.
        - :func:`pyzernike.zernike_polynomial` for the full Zernike polynomial computation.
        - The page :doc:`../../mathematical_description` in the documentation for the mathematical description of the Zernike polynomials.

    - ``rho`` and ``theta`` are expected to be floating point type numpy arrays of the same shape, dtype and in the range [0, 1] for ``rho``.
    - ``n``, ``m``, ``rho_derivative`` and ``theta_derivative`` are expected to be arrays of integers of the same length and valid values.

    The function is designed to precompute the useful terms for the Zernike polynomials, such as the powers of rho, the cosine and sine terms, and the logarithm of the factorials.

    Parameters
    ----------
    rho : numpy.ndarray (N-D array)
        The radial coordinate values with shape (...,) and floating point dtype corresponding to `float_type`.

    theta : numpy.ndarray (N-D array)
        The angular coordinate values with shape (...,) and floating point dtype corresponding to `float_type`. Must be None if `flag_radial` is True.

    n : numpy.array[numpy.integer]
        The orders of the Zernike polynomials to compute. Must be a 1D array of integers of type compatible with ``float_type``.

    m : numpy.array[numpy.integer]
        The azimuthal frequencies of the Zernike polynomials. Must be a 1D array of integers of type compatible with ``float_type``.

    numpy.array[numpy.integer]
        The orders of the derivatives with respect to rho. Must be a 1D array of integers of type compatible with ``float_type``.

    theta_derivative : Optional[numpy.array[numpy.integer]]
        The orders of the derivatives with respect to theta. Must be None if ``flag_radial`` is True. Otherwise, must be a 1D array of integers of type compatible with ``float_type``.

    flag_radial : bool
        If True, computes the sets for radial polynomials only (no angular part). The output sine and cosine frequency sets will be empty.
        If False, computes the sets for full Zernike polynomials (including angular part).

    precompute : bool
        If True, the useful terms for the Zernike polynomials are precomputed to optimize the computation.
        This is useful when computing multiple Zernike polynomials with the same `rho` and `theta` values.
        If False, the useful terms are computed on-the-fly for each polynomial, which may be slower but avoid memory overhead.

    float_type : type[numpy.floating]
        The floating point type used for the computations (e.g., numpy.float32, numpy.float64).

    Returns
    -------
    List[numpy.ndarray[numpy.floating]]
        A list of numpy.ndarray containing the Zernike polynomials for each (n, m, rho_derivative, theta_derivative) tuple, or the radial Zernike polynomials if `flag_radial` is True.
        Each polynomial has the shape of `rho` (and `theta` if `flag_radial` is False).

    """
    # Get the corresponding integer types
    int_type = core_corresponding_signed_integer_type(float_type)

    # Fast assertions on the inputs
    assert issubclass(float_type, numpy.floating), "[pyzernike-core] float_type must be a numpy floating point type."
    assert isinstance(n, numpy.ndarray) and n.ndim == 1 and numpy.issubdtype(n.dtype, int_type), "[pyzernike-core] n must be a 1D numpy array of integers of type compatible with float_type."
    assert isinstance(m, numpy.ndarray) and m.ndim == 1 and numpy.issubdtype(m.dtype, int_type), "[pyzernike-core] m must be a 1D numpy array of integers of type compatible with float_type."
    assert isinstance(rho_derivative, numpy.ndarray) and rho_derivative.ndim == 1 and numpy.issubdtype(rho_derivative.dtype, int_type), "[pyzernike-core] rho_derivative must be a 1D numpy array of integers of type compatible with float_type."
    assert isinstance(flag_radial, bool), "[pyzernike-core] flag_radial must be a boolean."
    assert flag_radial or (isinstance(theta_derivative, numpy.ndarray) and theta_derivative.ndim == 1 and numpy.issubdtype(theta_derivative.dtype, int_type)), "[pyzernike-core] theta_derivative must be a 1D numpy array of integers of type compatible with float_type when flag_radial is False."
    assert not flag_radial or theta_derivative is None, "[pyzernike-core] theta_derivative must be None when flag_radial is True."
    assert n.size == m.size == rho_derivative.size and (flag_radial or n.size == theta_derivative.size), "[pyzernike-core] n, m, rho_derivative and theta_derivative (if flag_radial is False) must have the same size."
    assert isinstance(rho, numpy.ndarray) and numpy.issubdtype(rho.dtype, float_type), "[pyzernike-core] rho must be a numpy array of floating point values of type compatible with float_type."
    assert flag_radial or (isinstance(theta, numpy.ndarray) and numpy.issubdtype(theta.dtype, float_type)), "[pyzernike-core] theta must be a numpy array of floating point values of type compatible with float_type when flag_radial is False."
    assert flag_radial or rho.shape == theta.shape, "[pyzernike-core] rho and theta must have the same shape when flag_radial is False."

    # Extract the shape
    shape = rho.shape

    # Create the output list
    output: List[numpy.ndarray] = []

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
    #     An array of shape=(max(m) + 1,) containing the indices of the cosine terms in `cosine_terms` for a given azimuthal frequency.
    #     This is used to map the computed angular polynomial coefficients to the precomputed cosine terms. ONLY USED IF `flag_radial` IS False.
    # 
    # cosine_terms : numpy.ndarray (2-D array)
    #     An array of shape=(..., Ncosine_terms) containing the cosine terms for the useful angular polynomials. ONLY USED IF `flag_radial` IS False.
    #   
    # sine_terms_indices_map : numpy.ndarray (1-D array)
    #     An array of shape=(max(m) + 1,) containing the indices of the sine terms in `sine_terms` for a given azimuthal frequency.
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
        precomputed_sets = core_create_precomputing_terms(n, m, rho_derivative, theta_derivative, flag_radial, float_type)
        powers_exponents: numpy.ndarray = precomputed_sets[0] # dtype=numpy.integer
        cosine_frequencies: numpy.ndarray = precomputed_sets[1] # dtype=numpy.integer
        sine_frequencies: numpy.ndarray = precomputed_sets[2] # dtype=numpy.integer
        factorials_integers: numpy.ndarray = precomputed_sets[3] # dtype=numpy.integer
        max_n: numpy.integer = precomputed_sets[4]
        max_m: numpy.integer = precomputed_sets[5]

        # Precompute the rho powers
        rho_powers_indices_map: numpy.ndarray = numpy.zeros(max_n + 1, dtype=int_type) # dtype=numpy.integer
        rho_powers_indices_map[powers_exponents] = numpy.arange(powers_exponents.size, dtype=int_type) # dtype=numpy.integer
        rho_powers: numpy.ndarray = numpy.power(rho[..., numpy.newaxis], powers_exponents, dtype=float_type) # dtype=numpy.floating
        assert rho_powers.dtype == float_type, "[pyzernike-core] Rho powers array has incorrect dtype."

        # Precompute the logarithm of the factorials
        log_factorials_indices_map: numpy.ndarray = numpy.zeros((max_n + 1,), dtype=int_type) # dtype=numpy.integer
        log_factorials_indices_map[factorials_integers] = numpy.arange(factorials_integers.size, dtype=int_type) # dtype=numpy.integer
        log_factorials: numpy.ndarray = gammaln(factorials_integers + 1).astype(float_type) # dtype=numpy.floating
        assert log_factorials.dtype == float_type, "[pyzernike-core] Log factorials array has incorrect dtype."

        # If flag_radial is True, we do not compute the angular terms
        if not flag_radial:

            # Precompute the cosine terms
            cosine_terms_indices_map: numpy.ndarray = numpy.zeros((max_m + 1,), dtype=int_type) # dtype=numpy.integer
            cosine_terms_indices_map[cosine_frequencies] = numpy.arange(cosine_frequencies.size, dtype=int_type) # dtype=numpy.integer
            cosine_terms: numpy.ndarray = numpy.cos(cosine_frequencies * theta[..., numpy.newaxis], dtype=float_type) # dtype=numpy.floating
            assert cosine_terms.dtype == float_type, "[pyzernike-core] Cosine terms array has incorrect dtype."

            # Precompute the sine terms
            sine_terms_indices_map: numpy.ndarray = numpy.zeros((max_m + 1,), dtype=int_type) # dtype=numpy.integer
            sine_terms_indices_map[sine_frequencies] = numpy.arange(sine_frequencies.size, dtype=int_type) # dtype=numpy.integer
            sine_terms: numpy.ndarray = numpy.sin(sine_frequencies * theta[..., numpy.newaxis], dtype=float_type) # dtype=numpy.floating
            assert sine_terms.dtype == float_type, "[pyzernike-core] Sine terms array has incorrect dtype."


    # =================================================================
    # Boucle over the polynomials to compute
    # =================================================================
    # Loop over the input lists to compute each polynomial
    for idx in range(len(n)):

        # Extract the n, m, rho_derivative
        n_idx: numpy.integer = int_type(n[idx])
        m_idx: numpy.integer = int_type(m[idx])
        abs_m_idx: numpy.integer = int_type(abs(m_idx))
        rho_derivative_idx: numpy.integer = int_type(rho_derivative[idx])

        # Case of n < 0, (n - m) is odd or |m| > n
        if n_idx < 0 or (n_idx - m_idx) % 2 != 0 or abs(m_idx) > n_idx:
            output.append(numpy.zeros(shape, dtype=float_type))
            continue

        # Case for derivatives of order greater than n_idx
        if rho_derivative_idx > n_idx:
            output.append(numpy.zeros(shape, dtype=float_type))
            continue

        # Case for radial polynomial and negative m
        if flag_radial and m_idx < 0:
            output.append(numpy.zeros(shape, dtype=float_type))
            continue

        # Compute the number of terms of the radial polynomial
        s: numpy.integer = min((n_idx - abs_m_idx) // 2, (n_idx - rho_derivative_idx) // 2)  # No computation for terms derivated more than the index of the polynomial
        k: numpy.ndarray = numpy.arange(0, s + 1, dtype=int_type)  # dtype=numpy.integer

        # Compute the coefficients of the radial polynomial
        if precompute:
            log_k_coef: numpy.ndarray = log_factorials[log_factorials_indices_map[n_idx - k]] - \
                                        log_factorials[log_factorials_indices_map[k]] - \
                                        log_factorials[log_factorials_indices_map[(n_idx + abs_m_idx) // 2 - k]] - \
                                        log_factorials[log_factorials_indices_map[(n_idx - abs_m_idx) // 2 - k]] # dtype=numpy.floating
        else:
            log_k_coef: numpy.ndarray = gammaln(n_idx - k + 1).astype(float_type) - \
                                        gammaln(k + 1).astype(float_type) - \
                                        gammaln((n_idx + abs_m_idx) // 2 - k + 1).astype(float_type) - \
                                        gammaln((n_idx - abs_m_idx) // 2 - k + 1).astype(float_type) # dtype=numpy.floating

        sign = 1 - 2 * (k % 2) # dtype=numpy.integer

        if rho_derivative_idx != 0:
            if precompute:
                log_k_coef += log_factorials[log_factorials_indices_map[n_idx - 2 * k]] - \
                              log_factorials[log_factorials_indices_map[n_idx - 2 * k - rho_derivative_idx]] # dtype=numpy.floating
            else:
                log_k_coef += gammaln(n_idx - 2 * k + 1).astype(float_type) - \
                              gammaln(n_idx - 2 * k - rho_derivative_idx + 1).astype(float_type) # dtype=numpy.floating
                
        # Assemble the coefficients
        coef: numpy.ndarray = numpy.multiply(sign, numpy.exp(log_k_coef, dtype=float_type), dtype=float_type)  # dtype=numpy.floating
        assert coef.dtype == float_type, "[pyzernike-core] Coefficients array has incorrect dtype."

        # Compute the rho power terms
        exponent: numpy.ndarray = n_idx - 2 * k - rho_derivative_idx # dtype=numpy.integer
        if precompute:
            rho_orders: numpy.ndarray = rho_powers[..., rho_powers_indices_map[exponent]] # dtype=numpy.floating
        else:
            rho_orders: numpy.ndarray = numpy.power(rho[..., numpy.newaxis], list(exponent)) # dtype=numpy.floating

        # Assemble the radial polynomial
        result: numpy.ndarray = numpy.tensordot(rho_orders, coef, axes=[[-1], [0]]) # dtype=numpy.floating

        # Theta part of the Zernike polynomial if flag_radial is False
        if not flag_radial:

            # Extract the angular theta derivative
            theta_derivative_idx: numpy.integer = theta_derivative[idx] # dtype=numpy.integer
            
            # According to the angular derivative, we compute the cosine factor
            if m_idx == 0:
                if theta_derivative_idx == 0:
                    cosine: numpy.ndarray = float_type(1.0)
                else:
                    cosine: numpy.ndarray = float_type(0.0)
                
            if m_idx > 0:
                if theta_derivative_idx == 0:
                    cosine: numpy.ndarray = cosine_terms[..., cosine_terms_indices_map[abs(m_idx)]] if precompute else numpy.cos(m_idx * theta, dtype=float_type)
                elif theta_derivative_idx % 4 == 0:
                    if precompute:
                        cosine: numpy.ndarray = (abs(m_idx) ** theta_derivative_idx) * cosine_terms[..., cosine_terms_indices_map[abs(m_idx)]]
                    else:
                        cosine: numpy.ndarray = (abs(m_idx) ** theta_derivative_idx) * numpy.cos(m_idx * theta, dtype=float_type)
                elif theta_derivative_idx % 4 == 1:
                    if precompute:
                        cosine: numpy.ndarray = - (abs(m_idx) ** theta_derivative_idx) * sine_terms[..., sine_terms_indices_map[abs(m_idx)]]
                    else:
                        cosine: numpy.ndarray = - (abs(m_idx) ** theta_derivative_idx) * numpy.sin(m_idx * theta, dtype=float_type)
                elif theta_derivative_idx % 4 == 2:
                    if precompute:
                        cosine: numpy.ndarray = - (abs(m_idx) ** theta_derivative_idx) * cosine_terms[..., cosine_terms_indices_map[abs(m_idx)]]
                    else:
                        cosine: numpy.ndarray = - (abs(m_idx) ** theta_derivative_idx) * numpy.cos(m_idx * theta, dtype=float_type)
                else:
                    if precompute:
                        cosine: numpy.ndarray = (abs(m_idx) ** theta_derivative_idx) * sine_terms[..., sine_terms_indices_map[abs(m_idx)]]
                    else:
                        cosine: numpy.ndarray = (abs(m_idx) ** theta_derivative_idx) * numpy.sin(m_idx * theta, dtype=float_type)

            if m_idx < 0:
                if theta_derivative_idx == 0:
                    cosine: numpy.ndarray = sine_terms[..., sine_terms_indices_map[abs(m_idx)]] if precompute else numpy.sin(abs(m_idx) * theta, dtype=float_type)
                elif theta_derivative_idx % 4 == 0:
                    if precompute:
                        cosine: numpy.ndarray = (abs(m_idx) ** theta_derivative_idx) * sine_terms[..., sine_terms_indices_map[abs(m_idx)]]
                    else:
                        cosine: numpy.ndarray = (abs(m_idx) ** theta_derivative_idx) * numpy.sin(abs(m_idx) * theta, dtype=float_type)
                elif theta_derivative_idx % 4 == 1:
                    if precompute:
                        cosine: numpy.ndarray = (abs(m_idx) ** theta_derivative_idx) * cosine_terms[..., cosine_terms_indices_map[abs(m_idx)]]
                    else:
                        cosine: numpy.ndarray = (abs(m_idx) ** theta_derivative_idx) * numpy.cos(abs(m_idx) * theta, dtype=float_type)
                elif theta_derivative_idx % 4 == 2:
                    if precompute:
                        cosine: numpy.ndarray = - (abs(m_idx) ** theta_derivative_idx) * sine_terms[..., sine_terms_indices_map[abs(m_idx)]]
                    else:
                        cosine: numpy.ndarray = - (abs(m_idx) ** theta_derivative_idx) * numpy.sin(abs(m_idx) * theta, dtype=float_type)
                else:
                    if precompute:
                        cosine: numpy.ndarray = - (abs(m_idx) ** theta_derivative_idx) * cosine_terms[..., cosine_terms_indices_map[abs(m_idx)]]
                    else:
                        cosine: numpy.ndarray = - (abs(m_idx) ** theta_derivative_idx) * numpy.cos(abs(m_idx) * theta, dtype=float_type)

            # Multiply the radial polynomial by the cosine factor
            cosine: numpy.ndarray = cosine.astype(float_type, copy=False)  # dtype=numpy.floating
    
            assert cosine.dtype == float_type, "[pyzernike-core] Cosine array has incorrect dtype."
            assert result.dtype == float_type, "[pyzernike-core] Radial polynomial array has incorrect dtype."

            numpy.multiply(result, cosine, dtype=float_type, out=result) # dtype=numpy.floating
            
            assert result.shape == rho.shape, "[pyzernike-core] Radial polynomial array has incorrect shape."

        # Save the polynomial
        output.append(result)

    return output


