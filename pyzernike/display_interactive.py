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

from .core.core_display import _core_display_interactive
from .core.parsing import _check_display_parameters


def display_interactive(*, precompute: bool = True, resolution: int = 200, rho_power: float = 1.0) -> None:
    r"""
    Explore the Zernike polynomials :math:`Z_{n}^{m}(\rho, \theta)` and the radial polynomials
    :math:`R_{n}^{m}(\rho)`, and their derivatives, in an interactive ``matplotlib`` figure.

    The left panel of the figure contains:

    - a selector to switch between the full Zernike polynomials and the radial polynomials,
    - text boxes to set the radial order :math:`n`, the azimuthal frequency :math:`m`, the order
      :math:`a` of the derivative with respect to :math:`\rho` and the order :math:`b` of the derivative
      with respect to :math:`\theta` (hidden for the radial polynomials),
    - text boxes to set the resolution of the grid and the exponent ``rho_power`` of the display mapping
      (see below),
    - a message line reporting the invalid inputs.

    The plot is updated when a value is submitted with ``Enter`` or when the selector is changed:

    - the radial polynomials are drawn as curves :math:`\rho \mapsto \frac{d^a R_{n}^{m}}{d\rho^a}(\rho)` on :math:`[0, 1]`,
    - the Zernike polynomials are drawn as colormaps of
      :math:`\frac{\partial^{a+b} Z_{n}^{m}}{\partial \rho^{a} \partial \theta^{b}}` on the unit disk.

    The vertical axis (radial) or the color scale (Zernike) spans :math:`[\min(-1, Z), \max(1, Z)]`.
    If :math:`|m| > n` or :math:`n - m` is odd, the displayed polynomial is zero and a warning is shown.

    .. seealso::

        - :func:`pyzernike.zernike_display` to display a given list of Zernike polynomials.
        - :func:`pyzernike.radial_display` to display a given list of radial polynomials.
        - :func:`pyzernike.zernike_polynomial` for the definition of the polynomials and of their derivatives.

    Parameters
    ----------
    precompute : bool, optional
        If True, the powers of :math:`\rho` and the cosine and sine terms are cached during the
        computation of each polynomial. Default is True.

    resolution : int, optional
        The initial number of points along each axis of the display grid. Default is 200.

    rho_power : float, optional
        The initial strictly positive exponent :math:`p` of the display mapping: at the displayed radial
        coordinate :math:`s \in [0, 1]`, the polynomial is evaluated at :math:`\rho = s^{p}`. With :math:`p < 1`
        (e.g. ``0.5``), the edge of the disk, where the high order polynomials oscillate the most, is
        enlarged; with :math:`p > 1`, the center is enlarged. Default is 1 (no mapping).

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If ``precompute`` is not a boolean, ``resolution`` is not an integer or ``rho_power`` is not a real number.

    ValueError
        If ``resolution`` is lower than 2 or ``rho_power`` is not strictly positive.

    Examples
    --------
    Open the interactive figure:

    .. code-block:: python

        from pyzernike import display_interactive

        display_interactive()

    Start with a finer grid and enlarge the edge of the disk to explore high orders:

    .. code-block:: python

        display_interactive(resolution=400, rho_power=0.5)

    """
    if not isinstance(precompute, bool):
        raise TypeError("precompute must be a boolean.")
    resolution, rho_power = _check_display_parameters(resolution, rho_power)

    _core_display_interactive(precompute, resolution, rho_power)