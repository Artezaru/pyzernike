from numbers import Integral
from typing import Sequence, List, Optional
import sympy

from .core_symbolic import core_symbolic

def zernike_symbolic(
    n: Sequence[Integral],
    m: Sequence[Integral],
    rho_derivative: Optional[Sequence[Integral]] = None,
    theta_derivative: Optional[Sequence[Integral]] = None,
    _skip: bool = False
) -> List[sympy.Expr]:
    r"""
    Compute the symbolic expression of the radial Zernike polynomial :math:`R_{n}^{m}(\rho)` for :math:`\rho \leq 1`.

    The Zernike polynomial is defined as follows:

    .. math::

        Z_{n}^{m}(\rho, \theta) = R_{n}^{m}(\rho) \cos(m \theta) \quad \text{if} \quad m > 0
    
    .. math::

        Z_{n}^{m}(\rho, \theta) = R_{n}^{-m}(\rho) \sin(-m \theta) \quad \text{if} \quad m < 0

    The derivative of order (derivative (a)) of the Zernike polynomial with respect to rho and order (derivative (b)) with respect to theta is defined as follows :

    .. math::

        \frac{\partial^{a}\partial^{b}Z_{n}^{m}(\rho, \theta)}{\partial \rho^{a} \partial \theta^{b}} = \frac{\partial^{a}R_{n}^{m}(\rho)}{\partial \rho^{a}} \frac{\partial^{b}\cos(m \theta)}{\partial \theta^{b}} \quad \text{if} \quad m > 0

    If :math:`|m| > n` or :math:`n < 0`, or :math:`(n - m)` is odd, the output is a zeros array with the same shape as :math:`\rho`.
    
    This function allows to compute several radial Zernike polynomials at once for different orders and degrees given as sequences. The output is a list of numpy arrays, each containing the expression of the radial Zernike polynomial for the corresponding order and degree.

    .. note::

        The symbol `x` is used to represent the radial coordinate :math:`\rho` in the symbolic expression.
        The symbol `y` is used to represent the angular coordinate :math:`\theta` in the symbolic expression.

    .. note::

        For developers, the ``_skip`` parameter is used to skip the checks for the input parameters. This is useful for internal use where the checks are already done.
        In this case :

        - ``n``, ``m``, ``rho_derivative`` and ``theta_derivative`` must be given as sequence of integers with the same length and valid values.

        See also :func:`pyzernike.core_symbolic` for more details on the input parameters.

    Parameters
    ----------
    n : Sequence[Integral]
        A list of the radial order(s) of the Zernike polynomial(s) to compute.

    m : Sequence[Integral]
        A list of the radial degree(s) of the Zernike polynomial(s) to compute.

    rho_derivative : Optional[Sequence[Integral]], optional
        A list of the order(s) of the radial derivative(s) to compute.
        If None, is it assumed that rho_derivative is 0 for all polynomials.

    theta_derivative : Optional[Sequence[Integral]], optional
        A list of the order(s) of the angular derivative(s) to compute.
        If None, is it assumed that theta_derivative is 0 for all polynomials.

    _skip : bool, optional
        If True, the checks for the input parameters are skipped. This is useful for internal use where the checks are already done.
        The default is False.

    Returns
    -------
    List[sympy.Expr]
        A list of symbolic expressions containing the Zernike polynomial values for each order and degree
        Each expression is a sympy expression that can be evaluated for specific values of :math:`\rho` and :math:`\theta`.

    Raises
    ------
    TypeError
        If the rho values are not a numpy array or if n and m are not integers.

    ValueError
        If the lengths of n and m are not the same.

    Examples
    --------
    Compute the expression of the radial Zernike polynomial :math:`Z_{2}^{0}(\rho, \theta)`:

    .. code-block:: python

        from pyzernike import zernike_symbolic
        result = zernike_symbolic(n=[2], m=[0])
        expression = result[0]  # result is a list, we take the first element

    Then evaluate the expression for a specific value of :math:`\rho` and :math:`\theta`:

    .. code-block:: python

        import numpy
        import sympy
        rho = numpy.linspace(0, 1, 100)
        theta = numpy.linspace(0, 2 * numpy.pi, 100)

        # `x` represents the radial coordinate in the symbolic expression
        # `y` represents the angular coordinate in the symbolic expression
        
        func = sympy.lambdify(['x', 'y'], expression, 'numpy')
        evaluated_result = func(rho, theta)

    """
    if not _skip:
        if not isinstance(n, Sequence) or not all(isinstance(i, Integral) for i in n):
            raise TypeError("n must be a sequence of integers.")
        if not isinstance(m, Sequence) or not all(isinstance(i, Integral) for i in m):
            raise TypeError("m must be a sequence of integers.")
        if rho_derivative is not None:
            if not isinstance(rho_derivative, Sequence) or not all(isinstance(i, Integral) and i >= 0 for i in rho_derivative):
                raise TypeError("rho_derivative must be a sequence of non-negative integers.")
        if theta_derivative is not None:
            if not isinstance(theta_derivative, Sequence) or not all(isinstance(i, Integral) and i >= 0 for i in theta_derivative):
                raise TypeError("theta_derivative must be a sequence of non-negative integers.")
        
        if len(n) != len(m):
            raise ValueError("n and m must have the same length.")
        if rho_derivative is not None and len(n) != len(rho_derivative):
            raise ValueError("n and rho_derivative must have the same length.")
        if theta_derivative is not None and len(n) != len(theta_derivative):
            raise ValueError("n and theta_derivative must have the same length.")
        if rho_derivative is None:
            rho_derivative = [0] * len(n)
        if theta_derivative is None:
            theta_derivative = [0] * len(n)

    # Compute the radial polynomials using the core_polynomial function
    zernike_expressions = core_symbolic(
        n=n,
        m=m,
        rho_derivative=rho_derivative,
        theta_derivative=theta_derivative,
        flag_radial=False
    )

    # Return the radial polynomials
    return zernike_expressions
