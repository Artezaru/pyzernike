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

from .core.parsing import _broadcast_integers, _check_display_parameters
from .core.core_display import _core_display


def zernike_display(
    n: Union[Integral, Sequence[Integral], numpy.ndarray],
    m: Union[Integral, Sequence[Integral], numpy.ndarray],
    rho_derivative: Optional[Union[Integral, Sequence[Integral], numpy.ndarray]] = 0,
    theta_derivative: Optional[Union[Integral, Sequence[Integral], numpy.ndarray]] = 0,
    *,
    precompute: bool = True,
    resolution: int = 200,
    rho_power: float = 1.0,
) -> None:
    r"""
    Display the Zernike polynomials :math:`Z_{n}^{m}(\rho, \theta)` and their derivatives in a ``matplotlib`` figure.

    Each polynomial is drawn as a colormap on the unit disk (polar axes). When several polynomials are
    given, the "Previous" and "Next" buttons of the figure navigate through them. The color scale spans
    :math:`[\min(-1, Z), \max(1, Z)]`.

    The Zernike polynomials [1]_ are defined on the unit disk for :math:`|m| \leq n` and :math:`n - m` even by:

    .. math::

        Z_{n}^{m}(\rho, \theta) = \begin{cases}
            R_{n}^{m}(\rho) \cos(m \theta) & \text{if } m \geq 0 \\
            R_{n}^{|m|}(\rho) \sin(|m| \theta) & \text{if } m < 0
        \end{cases}

    and their derivatives are given in :func:`pyzernike.zernike_polynomial`. If :math:`|m| > n`,
    :math:`n - m` is odd, :math:`a > n`, or :math:`m = 0` and :math:`b > 0`, the displayed polynomial is zero.

    **Shape of the inputs.** ``n``, ``m``, ``rho_derivative`` and ``theta_derivative`` can each be a
    single integer or a sequence of integers (list, tuple or 1D numpy array). The integers are repeated
    to the length of the sequences, which must all have the same length.

    .. seealso::

        - :func:`pyzernike.zernike_polynomial` for the numerical evaluation of the Zernike polynomials.
        - :func:`pyzernike.radial_display` to display the radial polynomials.
        - :func:`pyzernike.core._core_display` for the implementation of the display.

    Parameters
    ----------
    n : Union[Integral, Sequence[Integral], numpy.ndarray]
        The radial order(s) :math:`n`.

    m : Union[Integral, Sequence[Integral], numpy.ndarray]
        The azimuthal frequency(ies) :math:`m`.

    rho_derivative : Optional[Union[Integral, Sequence[Integral], numpy.ndarray]], optional
        The order(s) :math:`a \geq 0` of the derivative with respect to :math:`\rho`. Default is 0 (no derivative).
        None is accepted and means 0.

    theta_derivative : Optional[Union[Integral, Sequence[Integral], numpy.ndarray]], optional
        The order(s) :math:`b \geq 0` of the derivative with respect to :math:`\theta`. Default is 0 (no derivative).
        None is accepted and means 0.

    precompute : bool, optional
        If True, the powers of :math:`\rho` and the cosine and sine terms are cached and shared between
        the polynomials. Default is True.

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
        If ``n``, ``m``, ``rho_derivative`` or ``theta_derivative`` is not an integer or a sequence of
        integers, if ``precompute`` is not a boolean, or if ``resolution`` is not an integer or ``rho_power`` is not a real number.

    ValueError
        If the sequences do not have the same length or are empty, if a derivative order is negative, or if
        ``resolution`` is lower than 2 or ``rho_power`` is not strictly positive.

    References
    ----------
    .. [1] F. Zernike, "Beugungstheorie des Schneidenverfahrens und seiner verbesserten Form, der
       Phasenkontrastmethode," Physica 1(7-12), 689-704 (1934). https://doi.org/10.1016/S0031-8914(34)80259-5

    Examples
    --------
    Display the defocus :math:`Z_{2}^{0}(\rho, \theta)`:

    .. code-block:: python

        from pyzernike import zernike_display

        zernike_display(n=2, m=0)

    Display the first 15 polynomials in the OSA/ANSI ordering:

    .. code-block:: python

        from pyzernike import zernike_index_to_order

        n, m = zernike_index_to_order(range(15))
        zernike_display(n=n, m=m)

    .. image:: ../../../pyzernike/resources/zernike_display.png
        :width: 600px
        :align: center

    """
    if not isinstance(precompute, bool):
        raise TypeError("precompute must be a boolean.")
    resolution, rho_power = _check_display_parameters(resolution, rho_power)
    if rho_derivative is None:
        rho_derivative = 0
    if theta_derivative is None:
        theta_derivative = 0

    indices, _ = _broadcast_integers(n=n, m=m, rho_derivative=rho_derivative, theta_derivative=theta_derivative)
    if not indices["n"]:
        raise ValueError("At least one polynomial must be given.")
    if any(a < 0 for a in indices["rho_derivative"]):
        raise ValueError("rho_derivative must contain non-negative integers.")
    if any(b < 0 for b in indices["theta_derivative"]):
        raise ValueError("theta_derivative must contain non-negative integers.")

    _core_display(indices["n"], indices["m"], indices["rho_derivative"], indices["theta_derivative"], False, precompute, resolution, rho_power)