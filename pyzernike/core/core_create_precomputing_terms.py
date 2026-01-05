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

from typing import Set, Tuple, Optional
import numpy

from .core_corresponding_signed_integer_type import core_corresponding_signed_integer_type

def core_create_precomputing_terms(
    n: numpy.array,
    m: numpy.array,
    rho_derivative: numpy.array,
    theta_derivative: Optional[numpy.array],
    flag_radial: bool,
    float_type: type[numpy.floating]
) -> Tuple[numpy.array, numpy.array, numpy.array, numpy.array, numpy.integer, numpy.integer]:
    r"""
    Create the arrays of usefull exponents, frequencies and integers for the computation of Zernike polynomials and their derivatives.

    .. warning::

        This method is a core function of ``pyzernike`` that is not designed to be use by the users directly.
        Please use the high level functions.

    .. seealso::

        :func:`pyzernike.core.core_polynomial` for computing Zernike polynomials.

    For one defined Zernike polynomial of order ``n``, azimuthal frequency ``m`` and derivative with respect to rho ``a``, the usefull rho exponents are :

    .. math::

        \{ n - 2k - a \mid k = 0, 1, \ldots, \frac{n - |m|}{2} \}

    The useful integers for the factorials are :

    .. math::

        \{ n - k, k, \frac{n + |m|}{2} - k, \frac{n - |m|}{2} - k, n - 2k, n - 2k - a \mid k = 0, 1, \ldots, \frac{n - |m|}{2} \}

    if :math:`n \geq a` and :math:`n \geq |m|` and :math:`(n - m)` is even, otherwise the output is a zeros array with the same shape as :math:`\rho`.

    For the angular part, the usefull frequencies for the cosine and sine terms are :math:`|m|` depending on the parity of ``theta_derivative`` and the sign of ``m``.

    Parameters
    ----------
    n : numpy.array[numpy.integer]
        The orders of the Zernike polynomials to compute. Must be a 1D array of integers of type compatible with ``float_type``.

    m : numpy.array[numpy.integer]
        The azimuthal frequencies of the Zernike polynomials. Must be a 1D array of integers of type compatible with ``float_type``.

    rho_derivative : numpy.array[numpy.integer]
        The orders of the derivatives with respect to rho. Must be a 1D array of integers of type compatible with ``float_type``.

    theta_derivative : Optional[numpy.array[numpy.integer]]
        The orders of the derivatives with respect to theta. Must be None if ``flag_radial`` is True. Otherwise, must be a 1D array of integers of type compatible with ``float_type``.

    flag_radial : bool
        If True, computes the sets for radial polynomials only (no angular part). The output sine and cosine frequency sets will be empty.
        If False, computes the sets for full Zernike polynomials (including angular part).

    float_type : type[numpy.floating]
        The floating point type used for the computations (e.g., numpy.float32, numpy.float64). 

    Returns
    -------
    Tuple[numpy.array[numpy.integer], numpy.array[numpy.integer], numpy.array[numpy.integer], numpy.array[numpy.integer], numpy.integer, numpy.integer]
        A tuple containing:

        - ``powers_exponents``: An 1D array of unique integer exponents for the powers of rho needed for the computations with dtype compatible with ``float_type``.
        - ``cosine_frequencies``: An 1D array of unique integer frequencies for the cosine terms needed for the computations with dtype compatible with ``float_type``.
        - ``sine_frequencies``: An 1D array of unique integer frequencies for the sine terms needed for the computations with dtype compatible with ``float_type``.
        - ``factorials_integers``: An 1D array of unique integers for the factorials needed for the computations with dtype compatible with ``float_type``.
        - ``max_n``: The maximum order in ``n`` as integer of type compatible with ``float_type``.
        - ``max_abs_m``: The maximum absolute azimuthal frequency in ``m`` as integer of type compatible with ``float_type``.
    
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

    # Get the corresponding integer types
    int_type = core_corresponding_signed_integer_type(float_type)

    # Construct the sets for the useful terms
    powers_exponents_set: Set[numpy.integer] = set()
    cosine_frequencies_set: Set[numpy.integer] = set()
    sine_frequencies_set: Set[numpy.integer] = set()
    factorials_integers_set: Set[numpy.integer] = set()
    max_n: numpy.integer = int_type(0)
    max_abs_m: numpy.integer = int_type(0)

    for idx in range(len(n)):
        # Extract the n, m, dr values
        n_idx: numpy.integer = int_type(n[idx])
        m_idx: numpy.integer = int_type(m[idx])
        abs_m_idx: numpy.integer = int_type(abs(m_idx))
        dr_idx: numpy.integer = int_type(rho_derivative[idx])

        # Continue if n < dr or n < |m|
        if n_idx < dr_idx or n_idx < abs_m_idx:
            continue

        # Exponents and factorials sets
        max_k: numpy.integer = min((n_idx - abs_m_idx) // 2, (n_idx - dr_idx) // 2)

        for k in range(max_k + 1):
            k = int_type(k)
            powers_exponents_set.add(n_idx - 2 * k - dr_idx)
        
        for k in range(max_k + 1):
            k = int_type(k)
            factorials_integers_set.update([n_idx - k, k, (n_idx + abs_m_idx) // 2 - k, (n_idx - abs_m_idx) // 2 - k, n_idx - 2 * k, n_idx - 2 * k - dr_idx])
                                        
        # Cosine frequency and sine frequency sets
        if not flag_radial:
            # Extract the dt value
            dt_idx: numpy.integer = int_type(theta_derivative[idx])

            # Add frequencies in the cosine and sine terms sets
            if (m_idx > 0 and dt_idx % 2 == 0) or (m_idx < 0 and dt_idx % 2 == 1):
                cosine_frequencies_set.add(abs_m_idx)
            elif (m_idx < 0 and dt_idx % 2 == 0) or (m_idx > 0 and dt_idx % 2 == 1):
                sine_frequencies_set.add(abs_m_idx)

        # Updating the maximum values for n and m
        if n_idx > max_n:
            max_n: numpy.integer = n_idx
        if abs_m_idx > max_abs_m:
            max_abs_m: numpy.integer = abs_m_idx

    # Convert the sets to sorted numpy arrays
    powers_exponents_array: numpy.array = numpy.fromiter(powers_exponents_set, dtype=int_type) # dtype=numpy.integer
    cosine_frequencies_array: numpy.array = numpy.fromiter(cosine_frequencies_set, dtype=int_type) # dtype=numpy.integer
    sine_frequencies_array: numpy.array = numpy.fromiter(sine_frequencies_set, dtype=int_type) # dtype=numpy.integer
    factorials_integers_array: numpy.array = numpy.fromiter(factorials_integers_set, dtype=int_type) # dtype=numpy.integer

    return powers_exponents_array, cosine_frequencies_array, sine_frequencies_array, factorials_integers_array, max_n, max_abs_m
        