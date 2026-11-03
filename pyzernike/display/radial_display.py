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

from numbers import Integral
from typing import Optional, Sequence, Union

import numpy

from ..core.parsing import _broadcast_indices, _check_boolean, _check_display_parameters
from .core_display import _core_display


def radial_display(
    n: Union[Integral, Sequence[Integral], numpy.ndarray],
    m: Union[Integral, Sequence[Integral], numpy.ndarray],
    rho_derivative: Optional[Union[Integral, Sequence[Integral], numpy.ndarray]] = 0,
    *,
    precompute: bool = True,
    resolution: int = 200,
    rho_power: float = 1.0,
) -> None:
    r"""
    Display the radial Zernike polynomials :math:`R_{n}^{m}(\rho)` and their derivatives in a ``matplotlib`` figure.

    Each polynomial is drawn as a curve :math:`\rho \mapsto \frac{d^a R_{n}^{m}}{d\rho^a}(\rho)` on
    :math:`[0, 1]`. When several polynomials are given, the "Previous" and "Next" buttons of the figure
    navigate through them. The vertical axis spans :math:`[\min(-1, R), \max(1, R)]`.

    The radial Zernike polynomials [1]_ are defined for :math:`0 \leq m \leq n` and :math:`n - m` even by:

    .. math::

        R_{n}^{m}(\rho) = \sum_{k=0}^{\frac{n-m}{2}} \frac{(-1)^k (n-k)!}{k! \left(\frac{n+m}{2} - k\right)! \left(\frac{n-m}{2} - k\right)!} \rho^{n-2k}

    If :math:`m < 0`, :math:`m > n`, :math:`n - m` is odd or :math:`a > n`, the displayed polynomial is zero.

    **Shape of the inputs.** ``n``, ``m`` and ``rho_derivative`` can each be a single integer or a
    sequence of integers (list, tuple or 1D numpy array). The integers are repeated to the length of
    the sequences, which must all have the same length.

    .. seealso::

        - :func:`pyzernike.radial_polynomial` for the numerical evaluation of the radial polynomials.
        - :func:`pyzernike.zernike_display` to display the full Zernike polynomials.
        - :func:`pyzernike.display_interactive` to explore the polynomials in an interactive figure.
        - The page :doc:`../../mathematical_description/polynomials` for the radial polynomials and their derivatives.

    Parameters
    ----------
    n : Union[Integral, Sequence[Integral], numpy.ndarray]
        The radial order(s) :math:`n`.

    m : Union[Integral, Sequence[Integral], numpy.ndarray]
        The azimuthal frequency(ies) :math:`m`.

    rho_derivative : Optional[Union[Integral, Sequence[Integral], numpy.ndarray]], optional
        The order(s) :math:`a \geq 0` of the derivative with respect to :math:`\rho`. Default is 0 (no derivative).
        None is accepted and means 0.

    precompute : bool, optional
        If True, the powers of :math:`\rho` are cached and shared between the polynomials. Default is True.

    resolution : int, optional
        The number of points along each axis of the display grid. Default is 200.

    rho_power : float, optional
        The strictly positive exponent :math:`p` of the display mapping: at the displayed radial coordinate
        :math:`s \in [0, 1]`, the polynomial is evaluated at :math:`\rho = s^{p}`. With :math:`p < 1`
        (e.g. ``0.5``), the edge of the disk, where the high order polynomials oscillate the most, is
        enlarged; with :math:`p > 1`, the center is enlarged. Default is 1 (no mapping).

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If ``n``, ``m`` or ``rho_derivative`` is not an integer or a sequence of integers, if
        ``precompute`` is not a boolean, or if ``resolution`` is not an integer or ``rho_power`` is not a real number.

    ValueError
        If the sequences do not have the same length or are empty, if a derivative order is negative, or if
        ``resolution`` is lower than 2 or ``rho_power`` is not strictly positive.

    References
    ----------
    .. [1] F. Zernike, "Beugungstheorie des Schneidenverfahrens und seiner verbesserten Form, der
       Phasenkontrastmethode," Physica 1(7-12), 689-704 (1934). https://doi.org/10.1016/S0031-8914(34)80259-5

    Examples
    --------
    Display :math:`R_{2}^{0}(\rho)`:

    .. code-block:: python

        from pyzernike import radial_display

        radial_display(n=2, m=0)

    Display all the radial polynomials of order 4 and their first derivatives:

    .. code-block:: python

        radial_display(n=4, m=[0, 2, 4, 0, 2, 4], rho_derivative=[0, 0, 0, 1, 1, 1])

    .. image:: ../../../pyzernike/resources/radial_display.png
        :width: 600px
        :align: center

    """
    precompute = _check_boolean(precompute, "precompute")
    resolution, rho_power = _check_display_parameters(resolution, rho_power)
    indices, _ = _broadcast_indices(
        integers={"n": n, "m": m},
        orders={"rho_derivative": rho_derivative},
    )
    if not indices["n"]:
        raise ValueError("At least one polynomial must be given.")

    _core_display(
        n=indices["n"],
        m=indices["m"],
        rho_derivative=indices["rho_derivative"],
        theta_derivative=None,
        flag_radial=True,
        precompute=precompute,
        resolution=resolution,
        rho_power=rho_power,
    )