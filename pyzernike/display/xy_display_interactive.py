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

from numbers import Real
from typing import Optional

from ..core.parsing import _as_ellipse_parameters, _check_boolean, _check_resolution
from .core_xy_display import _core_xy_display_interactive


def xy_display_interactive(
    *,
    Rx: Optional[Real] = 1.0,
    Ry: Optional[Real] = 1.0,
    x0: Optional[Real] = 0.0,
    y0: Optional[Real] = 0.0,
    alpha: Optional[Real] = 0.0,
    h: Optional[Real] = 0.0,
    precompute: bool = True,
    resolution: int = 200,
) -> None:
    r"""
    Explore the Zernike polynomials on an elliptic annulus and their derivatives with respect to :math:`x` and
    :math:`y` in an interactive ``matplotlib`` figure.

    .. figure:: ../../../pyzernike/resources/elliptic_annulus_domain.png
        :width: 400px
        :align: center

        The parameters of the elliptic annulus domain.

    The left panel of the figure contains:

    - text boxes to set the radial order :math:`n`, the azimuthal frequency :math:`m` and the orders of the
      derivatives with respect to :math:`x` and :math:`y`,
    - text boxes to set the parameters :math:`(R_x, R_y, x_0, y_0, \alpha, h)` of the elliptic annulus
      (:math:`\alpha` in radians),
    - a text box to set the resolution of the grid,
    - a "Generate" button,
    - a message line reporting the invalid inputs.

    The plot is computed and updated only when the "Generate" button is clicked, so that the values can be
    typed without triggering a computation at each change. The polynomials are drawn as colormaps on a
    Cartesian grid covering the elliptic annulus, transparent outside the annulus, with the boundaries of the
    annulus. The color scale spans :math:`[\min(-1, Z), \max(1, Z)]`. If :math:`|m| > n` or :math:`n - m` is odd,
    the displayed polynomial is zero and a warning is shown.

    The same figure is opened from the command line with ``pyzernike --xy``.

    .. seealso::

        - :func:`pyzernike.xy_zernike_display` to display a given list of polynomials on the elliptic annulus.
        - :func:`pyzernike.xy_zernike_polynomial` for the definition of the polynomials and of their derivatives.
        - :func:`pyzernike.display_interactive` to explore the Zernike polynomials on the unit disk.
        - The page :doc:`../../mathematical_description/elliptic_annulus` for the extension of the Zernike polynomials to the elliptic annulus.

    Parameters
    ----------
    Rx : Optional[Real], optional
        The initial semi-axis :math:`R_x` of the ellipse along its x axis. Must be finite and strictly positive. Default is 1.0.

    Ry : Optional[Real], optional
        The initial semi-axis :math:`R_y` of the ellipse along its y axis. Must be finite and strictly positive. Default is 1.0.

    x0 : Optional[Real], optional
        The initial x coordinate :math:`x_0` of the center of the ellipse. Must be finite. Default is 0.0.

    y0 : Optional[Real], optional
        The initial y coordinate :math:`y_0` of the center of the ellipse. Must be finite. Default is 0.0.

    alpha : Optional[Real], optional
        The initial rotation angle :math:`\alpha` of the ellipse in radians. Must be finite. Default is 0.0.

    h : Optional[Real], optional
        The initial ratio :math:`h` of the inner to the outer semi-axes, in :math:`[0, 1)`. Default is 0.0 (full ellipse).

    precompute : bool, optional
        If True, the powers of :math:`\rho_{eq}` and :math:`1/r` and the cosine and sine terms are cached during
        the computation of each polynomial. Default is True.

    resolution : int, optional
        The initial number of points along each axis of the display grid. Default is 200.

    None is accepted for all the parameters of the ellipse, and means the default value.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If a parameter of the ellipse is not a real number, if ``precompute`` is not a boolean, or if
        ``resolution`` is not an integer.

    ValueError
        If a parameter of the ellipse is not finite, if ``Rx`` or ``Ry`` is not strictly positive, if ``h`` is
        not in :math:`[0, 1)`, or if ``resolution`` is lower than 2.

    Examples
    --------
    Open the interactive figure:

    .. code-block:: python

        from pyzernike import xy_display_interactive

        xy_display_interactive()

    Start with a rotated elliptic annulus:

    .. code-block:: python

        xy_display_interactive(Rx=2.0, Ry=1.0, alpha=0.5, h=0.3)

    """
    Rx, Ry, x0, y0, alpha, h = _as_ellipse_parameters(Rx, Ry, x0, y0, alpha, h)
    precompute = _check_boolean(precompute, "precompute")
    resolution = _check_resolution(resolution)

    _core_xy_display_interactive(
        Rx=Rx,
        Ry=Ry,
        x0=x0,
        y0=y0,
        alpha=alpha,
        h=h,
        precompute=precompute,
        resolution=resolution,
    )