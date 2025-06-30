from numbers import Integral
from typing import Sequence, Optional

from .core_display import core_display

def radial_display(
    n: Sequence[Integral],
    m: Sequence[Integral],
    rho_derivative: Optional[Sequence[Integral]] = None,
    _skip: bool = False
) -> None:
    r"""
    Display the radial Zernike polynomial :math:`R_{n}^{m}(\rho)` for :math:`\rho \leq 1` in an interactive matplotlib figure.

    The radial Zernike polynomial is defined as follows:

    .. math::

        R_{n}^{m}(\rho) = \sum_{k=0}^{(n-m)/2} \frac{(-1)^k (n-k)!}{k! ((n+m)/2 - k)! ((n-m)/2 - k)!} \rho^{n-2k}

    The derivative of order (derivative (a)) of the radial Zernike polynomial is defined as follows:

    .. math::

        \frac{d^{a}R_{n}^{m}(\rho)}{d\rho^{a}} = \sum_{k=0}^{(n-m)/2} \frac{(-1)^k (n-k)!}{k! ((n+m)/2 - k)! ((n-m)/2 - k)!} (n-2k) (n-2k-1) \ldots (n-2k-a+1) \rho^{n-2k-a}
    
    This function allows to display several radial Zernike polynomials at once for different orders and degrees given as sequences.

    .. note::

        For developers, the ``_skip`` parameter is used to skip the checks for the input parameters. This is useful for internal use where the checks are already done.
        In this case :

        - ``n``, ``m`` and ``rho_derivative`` must be given as sequence of integers with the same length and valid values.

        See also :func:`pyzernike.core_display` for more details on the input parameters.

    .. seealso::

        :func:`pyzernike.zernike_display` for displaying the full Zernike polynomial.

    Parameters
    ----------
    n : Sequence[Integral]
        A list of the radial order(s) of the Zernike polynomial(s) to display.

    m : Sequence[Integral]
        A list of the radial degree(s) of the Zernike polynomial(s) to display.

    rho_derivative : Optional[Sequence[Integral]], optional
        A list of the order(s) of the radial derivative(s) to display.
        If None, is it assumed that rho_derivative is 0 for all polynomials.

    _skip : bool, optional
        If True, the checks for the input parameters are skipped. This is useful for internal use where the checks are already done.
        The default is False.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If the rho values are not a numpy array or if n and m are not integers.

    ValueError
        If the lengths of n and m are not the same.

    Examples
    --------
    Display the radial Zernike polynomial :math:`R_{2}^{0}(\rho)`:

    .. code-block:: python

        from pyzernike import radial_display
        radial_display(n=[2], m=[0]) # This will display the radial Zernike polynomial R_2^0 in an interactive matplotlib figure.

    To display multiple radial Zernike polynomials, you can pass sequences for `n` and `m`:

    .. code-block:: python

        from pyzernike import radial_display
        radial_display(n=[2, 3, 4], m=[0, 1, 2], rho_derivative=[0, 0, 1])

    .. image:: ../../../pyzernike/resources/radial_display.png
        :width: 600px
        :align: center

    """
    if not _skip:
        if not isinstance(n, Sequence) or not all(isinstance(i, Integral) for i in n):
            raise TypeError("n must be a sequence of integers.")
        if not isinstance(m, Sequence) or not all(isinstance(i, Integral) for i in m):
            raise TypeError("m must be a sequence of integers.")
        if rho_derivative is not None:
            if not isinstance(rho_derivative, Sequence) or not all(isinstance(i, Integral) and i >= 0 for i in rho_derivative):
                raise TypeError("rho_derivative must be a sequence of non-negative integers.")
        
        if len(n) != len(m):
            raise ValueError("n and m must have the same length.")
        if rho_derivative is not None and len(n) != len(rho_derivative):
            raise ValueError("n and rho_derivative must have the same length.")
        if rho_derivative is None:
            rho_derivative = [0] * len(n)

    # Compute the radial polynomials using the core_polynomial function
    core_display(
        n=n,
        m=m,
        rho_derivative=rho_derivative,
        theta_derivative=None,
        flag_radial=True
    )
