from typing import Sequence, List
from numbers import Integral


def zernike_order_to_index(n : Sequence[Integral], m : Sequence[Integral], _skip: bool = False) -> List[int]:
    r"""
    Convert Zernike orders (n, m) to their corresponding indices in the OSA/ANSI Zernike polynomial ordering.

    .. math::

        j = \frac{n(n + 2) + m}{2}

    If :math:`|m| > n` or :math:`n < 0`, or :math:`(n - m)` is odd, an error is raised.

    .. seealso::

        - :func:`pyzernike.zernike_polynomial` for computing the Zernike polynomial :math:`Z_{n}^{m}(\rho, \theta)`.
        - :func:`pyzernike.zernike_index_to_order` for converting indices back to Zernike orders.

    .. note::

        For developers, the ``_skip`` parameter is used to skip the checks for the input parameters. This is useful for internal use where the checks are already done.
        In this case :

        - ``n`` and ``m`` must be given as sequence of integers with the same length and valid values.

    Parameters
    ----------
    n : Sequence[Integral]
        The radial orders of the Zernike polynomials.

    m : Sequence[Integral]
        The azimuthal orders of the Zernike polynomials.

    _skip : bool, optional
        If True, skips input validation checks. Default is False.

    Returns
    -------
    List[int]
        A list of indices corresponding to the Zernike orders.

    Raises
    ------
    TypeError
        If `n` or `m` are not sequences of integers.

    ValueError
        If `n` and `m` do not have the same length.

    Examples
    --------

    .. code-block:: python

        from pyzernike import zernike_order_to_index

        n = [2, 3, 4]
        m = [0, 1, -2]
        indices = zernike_order_to_index(n, m)
        print(indices)  # Output: [4, 8, 11]

    """
    if not _skip:
        if not isinstance(n, Sequence) or not all(isinstance(i, Integral) for i in n):
            raise TypeError("n must be a sequence of integers.")
        if not isinstance(m, Sequence) or not all(isinstance(i, Integral) for i in m):
            raise TypeError("m must be a sequence of integers.")

        if len(n) != len(m):
            raise ValueError("n and m must have the same length.")
        
        if any( i < 0 or abs(j) > i or (i - abs(j)) % 2 != 0 for i, j in zip(n, m)):
            raise ValueError("Invalid Zernike orders: |m| must be <= n, n must be non-negative, and (n - |m|) must be even.")

    return [int((i * (i + 2) + j) / 2) for i, j in zip(n, m)]
