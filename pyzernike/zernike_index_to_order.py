from typing import Sequence, List, Tuple
from numbers import Integral


def zernike_index_to_order(j: Sequence[Integral], _skip: bool = False) -> Tuple[List[int], List[int]]:
    r"""
    Convert indices in the OSA/ANSI Zernike polynomial ordering to their corresponding Zernike orders (n, m).

    .. math::

        j = \frac{n(n + 2) + m}{2}

    ``j`` must be a sequence of non-negative integers.

    The process to compute the Zernike orders from the index is as follows:

    .. math::

        n(n+2) = 2j - m \in [2j - n, 2j + n]

    So :

    .. math::
    
        n = \text{int}\left(\frac{-1 + \sqrt{1 + 8j}}{2}\right) \quad \text{and} \quad m = 2j - n(n + 2)


    .. seealso::

        - :func:`pyzernike.zernike_polynomial` for computing the Zernike polynomial :math:`Z_{n}^{m}(\rho, \theta)`.
        - :func:`pyzernike.zernike_order_to_index` for converting Zernike orders to indices.

    .. note::

        For developers, the ``_skip`` parameter is used to skip the checks for the input parameters. This is useful for internal use where the checks are already done.
        In this case :

        - ``j`` must be given as sequence of integers with valid values.

    Parameters
    ----------
    j : Sequence[Integral]
        The indices of the Zernike polynomials in the OSA/ANSI ordering.

    _skip : bool, optional
        If True, skips input validation checks. Default is False.

    Returns
    -------
    List[int]
        A list of radial orders (n) of the Zernike polynomials.

    List[int]
        A list of azimuthal orders (m) of the Zernike polynomials.

    Raises
    ------
    TypeError
        If `j` is not a sequence of integers.

    Examples
    --------

    .. code-block:: python

        from pyzernike import zernike_index_to_order

        j = [2, 3, 4]
        n, m = zernike_index_to_order(j)
        print(n)  # Output: [1, 2, 2]
        print(m)  # Output: [1, -2, 0]

    """
    if not _skip:
        if not isinstance(j, Sequence) or not all(isinstance(i, Integral) for i in j):
            raise TypeError("j must be a sequence of integers.")
        
        if any(i < 0 for i in j):
            raise ValueError("j must be non-negative integers.")

    n = [int((-1 + (1 + 8 * i) ** 0.5) / 2) for i in j]
    m = [2 * i - n_i * (n_i + 2) for n_i, i in zip(n, j)]

    return n, m