import numpy
import numbers
from typing import Sequence, List, Optional
from scipy.special import gammaln

def common_zernike_polynomial(rho: numpy.ndarray, theta: numpy.ndarray, n: Sequence[int], m: Sequence[int], rho_derivative: Optional[Sequence[int]] = None, theta_derivative: Optional[Sequence[int]] = None, default: float = numpy.nan, _skip: bool = False) -> List[numpy.ndarray]:
    r"""
    Computes the radial Zernike polynomial :math:`R_{n}^{m}(\rho)` for :math:`\rho \leq 1`.

    The radial Zernike polynomial is defined as follows:

    .. math::

        R_{n}^{m}(\rho) = \sum_{k=0}^{(n-m)/2} \frac{(-1)^k (n-k)!}{k! ((n+m)/2 - k)! ((n-m)/2 - k)!} \rho^{n-2k}

    if :math:`n < 0`, :math:`m < 0`, :math:`n < m`, or :math:`(n - m)` is odd, the output is a zeros array with the same shape as :math:`\rho`.

    The derivative of order (derivative (a)) of the radial Zernike polynomial is defined as follows :

    .. math::

        \frac{d^{a}R_{n}^{m}(\rho)}{d\rho^{a}} = \sum_{k=0}^{(n-m)/2} \frac{(-1)^k (n-k)!}{k! ((n+m)/2 - k)! ((n-m)/2 - k)!} (n-2k) (n-2k-1) \ldots (n-2k-a+1) \rho^{n-2k-a}

    The computation of the factorial is done using the function :func:`scipy.special.gammaln` for better performance and stability.

    .. math::

        \text{log}(n!) = \text{gammaln}(n+1)

    So the coefficient of the radial polynomial is computed as follows:

    .. math::

        \frac{(n-k)!}{k! ((n+m)/2 - k)! ((n-m)/2 - k)!} = \text{exp} (\text{gammaln}(n-k+1) - \text{gammaln}(k+1) - \text{gammaln}((n+m)/2 - k + 1) - \text{gammaln}((n-m)/2 - k + 1))
    
    if :math:`\rho` is not in :math:`0 \leq \rho \leq 1` or :math:`\rho` is numpy.nan, the output is set to the default value (numpy.nan by default).

    This function allows to compute several radial Zernike polynomials at once for different orders and degrees, which can be more efficient than calling the radial polynomial function multiple times.
    The :math:`\rho` values are the same for all the polynomials, and the orders and degrees are provided as sequences.

    .. note::

        This method precompute all :math:`\rho^{p}` for :math:`p < \text{max}(n)` and all factorials :math:`p!`.
        This method is usefull if the user want to compute many various couples ``(n, m, rho_derivative)``, otherwize the function
        :func:`pyzernike.radial_polynomial` is better.

    .. note::

        An alias for this method is ``optR``.

    Parameters
    ----------
    rho : numpy.ndarray
        The rho values.
    
    n : Sequence[int]
        A sequence (list/tuple) of integers containing the order `n` of each radial Zernike polynomials to compute.

    m : Sequence[int]
        A sequence (list/tuple) of integers containing the degree `m` of each radial Zernike polynomials to compute.

    rho_derivative : Optional[Sequence[int]], optional
        A sequence (list/tuple) of integers containing the order of the derivative with respect to ``rho`` for of each radial Zernike polynomials to compute.
        If None, the rho_derivative is a sequence of zeros. Default is None.

    default : float, optional
        The default value for invalid rho values. The default is numpy.nan.
    
    Returns
    -------
    List[numpy.ndarray]
        A list of radial Zernike polynomials corresponding to each `(n, m, rho_derivative)`.
    
    Raises
    ------
    TypeError
        If the rho values are not a numpy array or if n and m are not sequences of integers with the same length.
        If the rho_derivative is not a sequence of integers with the length as n and m.
    ValueError
        If the rho_derivative is negative.

    Examples
    --------
    
    .. code-block:: python

        import numpy
        from pyzernike import optimized_global_radial_polynomial
        rho = numpy.linspace(0, 1, 100)
        optimized_global_radial_polynomial(rho, n=[2], m=[0])

    returns a list containing the one element : the radial Zernike polynomial :math:`R_{2}^{0}(\rho)` for :math:`\rho \leq 1`.

    .. code-block:: python

        import numpy
        from pyzernike import optimized_global_radial_polynomial
        rho = numpy.linspace(0, 1, 100)
        optimized_global_radial_polynomial(rho, n=[2,2], m=[0,0], rho_derivative=[0,1])

    returns a list containing the two elements : the radial Zernike polynomial :math:`R_{2}^{0}(\rho)` and its first derivative with respect to rho.
    """
    if not _skip:
        # Check the input parameters
        if not isinstance(rho, numpy.ndarray):
            raise TypeError("Rho values must be a numpy array.")
        if not isinstance(n, Sequence) or not isinstance(m, Sequence):
            raise TypeError("n and m must be sequences (list / tuple)")
        if not len(n) == len(m):
            raise ValueError("n and m must be sequences (list / tuple) with the same length.")
        if not all(isinstance(item, numbers.Integral) for item in n) or not all(isinstance(item, numbers.Integral) for item in m):
            raise ValueError("n and m must be sequences (list / tuple) of intergers.")
        
        if rho_derivative is None:
            rho_derivative = [0 for _ in range(len(n))]
        if not isinstance(rho_derivative, Sequence):
            raise TypeError("rho_derivative must be a sequence (list / tuple)")
        if not len(rho_derivative) == len(m):
            raise ValueError("rho_derivative must be a sequence (list / tuple) with the same length as n and m.")
        if not all(isinstance(item, numbers.Integral) for item in rho_derivative) or not all(item >= 0 for item in rho_derivative):
            raise ValueError("rho_derivative must be a sequence (list / tuple) of positives intergers.")
        if not isinstance(default, numbers.Real):
            raise TypeError("The default value must be a real number.")
        
        if theta_derivative is None:
            theta_derivative = [0 for _ in range(len(n))]
        if not isinstance(theta_derivative, Sequence):
            raise TypeError("theta_derivative must be a sequence (list / tuple)")
        if not len(theta_derivative) == len(m):
            raise ValueError("theta_derivative must be a sequence (list / tuple) with the same length as n and m.")
        if not all(isinstance(item, numbers.Integral) for item in theta_derivative) or not all(item >= 0 for item in theta_derivative):
            raise ValueError("theta_derivative must be a sequence (list / tuple) of positives intergers.")
        if not isinstance(default, numbers.Real):
            raise TypeError("The default value must be a real number.")
        
        list_n = list(n)
        list_m = list(m)
        list_rho_derivative = list(rho_derivative)
        list_theta_derivative = list(theta_derivative)

        # flatten the rho values to handle multiple dimensions
        shape = rho.shape
        rho_flat = rho.flatten()
        theta_flat = theta.flatten()

        # Create the mask for valid rho values
        unit_circle_mask = numpy.logical_and(0 <= rho_flat, rho_flat <= 1)
        nan_mask = numpy.logical_or(numpy.isnan(rho_flat), numpy.isnan(theta_flat))
        valid_mask = numpy.logical_and(unit_circle_mask, ~nan_mask)

        # Compute for valid rho values
        rho_valid = rho_flat[valid_mask]
        theta_valid = theta_flat[valid_mask]

        # Initialize the output array
        output_flat = [numpy.full_like(rho_flat, default) for _ in range(len(list_n))]

    if _skip:
        list_n = n
        list_m = m
        if rho_derivative is None:
            list_rho_derivative = [0 for _ in range(len(list_n))]
        else:
            list_rho_derivative = rho_derivative
        if theta_derivative is None:
            list_theta_derivative = [0 for _ in range(len(list_n))]
        else:
            list_theta_derivative = theta_derivative
        rho_flat = rho_valid = rho
        theta_flat = theta_valid = theta
        output_flat = [None for _ in range(len(list_n))]

    # Compute the interesting values
    max_n = max(list_n)
    max_m = max(list_m)
    rho_powers_precomputed = numpy.power(rho_valid[:, numpy.newaxis], numpy.arange(max_n+1)) # Shape(Npoints, Max_n + 1)
    log_fact_precomputed = gammaln(numpy.arange(max_n+1) + 1) # Shape(Max_n + 1,)
    cosine_precomputed = numpy.cos(numpy.arange(max_m+1) * theta_valid[:, numpy.newaxis]) # Shape(Npoints, Max_m +1)
    sine_precomputed = numpy.sin(numpy.arange(max_m+1) * theta_valid[:, numpy.newaxis]) # Shape(Npoints, Max_m +1)

    # Computing the radial polynomials for each (n, m, rho_derivative)
    for idx in range(len(list_n)):
        # Extract the n, m, rho_derivative
        n = list_n[idx]
        m = list_m[idx]
        rho_derivative = list_rho_derivative[idx]
        theta_derivative = list_theta_derivative[idx]

        # Case of n < 0, (n - m) is odd or |m| > n
        if n < 0 or (n - m) % 2 != 0 or abs(m) > n:
            if not _skip:
                output_flat[idx][valid_mask] = 0.0
                output_flat[idx] = output_flat[idx].reshape(shape)
            if _skip:
                output_flat[idx] = numpy.zeros_like(rho_flat)
            continue

        # Compute the number of terms
        s = (n-m)//2
        k = numpy.arange(0, s+1)

        # Vectorize coefficients
        log_k_coef = log_fact_precomputed[n - k] - log_fact_precomputed[k] - log_fact_precomputed[(n + m) // 2 - k] - log_fact_precomputed[(n - m) // 2 - k]
        sign = 1 - 2 * (k % 2)
    
        if rho_derivative != 0:
            second_member = numpy.where(n - 2 * k - rho_derivative >= 0, n - 2 * k - rho_derivative, 0)
            log_k_coef += log_fact_precomputed[n - 2 * k] - log_fact_precomputed[second_member]

        coef = sign * numpy.exp(log_k_coef)

        # Compute the rho power
        exponent = n - 2 * k - rho_derivative
        exponent_positive_mask = exponent > 0
        exponent_0_mask = exponent == 0
        exponent_negative_mask = exponent < 0

        rho_powers = numpy.empty((len(rho_valid), len(k)))
        rho_powers[:, exponent_positive_mask] = rho_powers_precomputed[:, exponent[exponent_positive_mask]]
        rho_powers[:, exponent_0_mask] = 1.0
        rho_powers[:, exponent_negative_mask] = 0.0

        # Compute the phase
        if m == 0:
            if theta_derivative == 0:
                cosine_valid = 1.0
            else:
                cosine_valid = 0.0
            
        if m > 0:
            if theta_derivative == 0:
                cosine_valid = cosine_precomputed[:,m]
            elif theta_derivative % 4 == 0:
                cosine_valid = (m ** theta_derivative) * cosine_precomputed[:,m]
            elif theta_derivative % 4 == 1:
                cosine_valid = - (m ** theta_derivative) * sine_precomputed[:,m]
            elif theta_derivative % 4 == 2:
                cosine_valid = - (m ** theta_derivative) * cosine_precomputed[:,m]
            else:
                cosine_valid = (m ** theta_derivative) * sine_precomputed[:,m]
            
        if m < 0:
            if theta_derivative == 0:
                cosine_valid = sine_precomputed[:,m]
            elif theta_derivative % 4 == 0:
                cosine_valid = (m ** theta_derivative) * sine_precomputed[:,m]
            elif theta_derivative % 4 == 1:
                cosine_valid = (m ** theta_derivative) * cosine_precomputed[:,m]
            elif theta_derivative % 4 == 2:
                cosine_valid = - (m ** theta_derivative) * sine_precomputed[:,m]
            else:
                cosine_valid = - (m ** theta_derivative) * cosine_precomputed[:,m]

        # Compute the result
        result = cosine_valid * numpy.dot(rho_powers, coef)

        if not _skip:
            # Assign the result to the output
            output_flat[idx][valid_mask] = result
            
            # Reshape the output to the original shape
            output_flat[idx] = output_flat[idx].reshape(shape)  

        if _skip:
            output_flat[idx] = result   

    return output_flat # Here it is all reshape
