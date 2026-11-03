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

from numbers import Integral, Real
from typing import List, Optional, Sequence, Union

import numpy

from ..core.parsing import (
    _as_choices,
    _as_common_floating_arrays,
    _as_default_value,
    _as_ellipse_parameters,
    _as_integers,
    _broadcast_parsed,
    _check_boolean,
    _format_output,
)
from ..core.core_cartesian_to_elliptic_annulus import _core_cartesian_to_elliptic_annulus
from ..core.core_xy_polynomial import ELLIPSE_PARAMETERS, _core_xy_parameter_derivatives


def xy_zernike_parameter_derivative(
    x: numpy.ndarray,
    y: numpy.ndarray,
    n: Union[Integral, Sequence[Integral], numpy.ndarray],
    m: Union[Integral, Sequence[Integral], numpy.ndarray],
    parameter: Union[str, Sequence[str], numpy.ndarray],
    Rx: Optional[Real] = 1.0,
    Ry: Optional[Real] = 1.0,
    x0: Optional[Real] = 0.0,
    y0: Optional[Real] = 0.0,
    alpha: Optional[Real] = 0.0,
    h: Optional[Real] = 0.0,
    default: Union[int, float] = numpy.nan,
    *,
    precompute: bool = True,
    normalize: bool = False,
    to_list: bool = True,
) -> Union[numpy.ndarray, List[numpy.ndarray]]:
    r"""
    Compute the first order derivatives of the Zernike polynomials on an elliptic annulus [1]_ with respect to
    the parameters :math:`(x_0, y_0, R_x, R_y, \alpha, h)` of the annulus, at fixed Cartesian coordinates
    :math:`(x, y)`.

    .. figure:: ../../../pyzernike/resources/elliptic_annulus_domain.png
        :width: 400px
        :align: center

        The parameters of the elliptic annulus domain.

    The Zernike polynomials on the elliptic annulus are :math:`Zxy_{n}^{m}(x, y) = Z_{n}^{m}\left(\rho_{eq}, \theta_{eq}\right)`
    (see :func:`pyzernike.xy_zernike_polynomial`). These derivatives are used to refine the domain of
    definition, e.g. to fit the parameters of the annulus with a Gauss-Newton algorithm: for a surface
    :math:`W = \sum_j c_j Zxy_{j}`, :math:`\frac{\partial W}{\partial p} = \sum_j c_j \frac{\partial Zxy_{j}}{\partial p}`.
    The residual and Jacobian functions of such a fit are directly given by :func:`pyzernike.xy_zernike_least_squares`.

    Let :math:`u = X / R_x` and :math:`v = Y / R_y`, where :math:`X = c (x - x_0) + s (y - y_0)` and
    :math:`Y = -s (x - x_0) + c (y - y_0)` are the coordinates in the frame of the ellipse
    (:math:`c = \cos\alpha`, :math:`s = \sin\alpha`). The polynomial depends on :math:`(x_0, y_0, R_x, R_y, \alpha)`
    only through :math:`(u, v)`, so with :math:`Z_u = \partial_u Zxy_{n}^{m}` and :math:`Z_v = \partial_v Zxy_{n}^{m}`:

    .. math::

        \frac{\partial Zxy_{n}^{m}}{\partial x_0} = -\frac{c}{R_x} Z_u + \frac{s}{R_y} Z_v
        \quad \text{and} \quad
        \frac{\partial Zxy_{n}^{m}}{\partial y_0} = -\frac{s}{R_x} Z_u - \frac{c}{R_y} Z_v

    .. math::

        \frac{\partial Zxy_{n}^{m}}{\partial R_x} = -\frac{u}{R_x} Z_u
        \quad \text{and} \quad
        \frac{\partial Zxy_{n}^{m}}{\partial R_y} = -\frac{v}{R_y} Z_v
        \quad \text{and} \quad
        \frac{\partial Zxy_{n}^{m}}{\partial \alpha} = \frac{R_y}{R_x} v Z_u - \frac{R_x}{R_y} u Z_v

    The parameter :math:`h` appears only in :math:`\rho_{eq} = \frac{r - h}{1 - h}` (:math:`r = \sqrt{u^2 + v^2}`), so:

    .. math::

        \frac{\partial Zxy_{n}^{m}}{\partial h} = \frac{\rho_{eq} - 1}{1 - h} \frac{\partial Z_{n}^{m}}{\partial \rho}(\rho_{eq}, \theta_{eq})

    :math:`Z_u` and :math:`Z_v` are computed as in :func:`pyzernike.xy_zernike_polynomial`, without singularity
    at the center of the ellipse.

    - If :math:`|m| > n` or :math:`n - m` is odd, the derivative is zero.
    - Where the point is outside the elliptic annulus (:math:`\rho_{eq}` outside :math:`[0, 1]`), or the
      coordinates are not finite, the output is set to ``default``.

    .. note::

        The derivatives are those of the value of the polynomial at a fixed point inside the annulus. The
        domain itself depends on the parameters: when they change, points may enter or leave the annulus,
        which is not differentiable. During an optimization, keep the set of points fixed within an iteration
        (or discard the points close to the boundary), and keep :math:`h` in :math:`[0, 1)` (the derivative
        with respect to :math:`h` at :math:`h = 0` is the right derivative). The parameters can not be estimated
        together with the coefficients for a full ellipse (see the page :doc:`../../mathematical_description/fitting`).

    **Shape of the inputs and of the output.** ``n``, ``m`` and ``parameter`` can each be a single value or
    a sequence (list, tuple or 1D numpy array):

    - If all of them are single values, a single derivative is computed and returned as an array with the
      shape of ``x``.
    - If at least one of them is a sequence, the single values are repeated to the length :math:`N` of the
      sequences (which must all have the same length), and :math:`N` derivatives are computed. They are
      returned as a list of :math:`N` arrays with the shape of ``x`` (``to_list=True``), or as a single
      array of shape :math:`(N, \dots)` (``to_list=False``).

    .. seealso::

        - :func:`pyzernike.xy_zernike_polynomial` to compute the Zernike polynomials on the elliptic annulus and
          their derivatives with respect to :math:`x` and :math:`y`.
        - :func:`pyzernike.cartesian_to_elliptic_annulus` to compute the equivalent polar coordinates.
        - :func:`pyzernike.xy_zernike_least_squares` for the residual and Jacobian functions of the fit of the
          parameters of the annulus.
        - The page :doc:`../../mathematical_description/elliptic_annulus` for the derivatives with respect to the parameters of the annulus.
        - The page :doc:`../../mathematical_description/fitting` for the estimation of the parameters of the annulus.

    Parameters
    ----------
    x : numpy.ndarray
        The x Cartesian coordinates with any shape ``(...)``.

    y : numpy.ndarray
        The y Cartesian coordinates with the same shape as ``x``.
        If ``x`` and ``y`` are floating point arrays of the same dtype (e.g. ``numpy.float32``), this
        dtype is kept. Otherwise, both are converted to ``numpy.float64``.

    n : Union[Integral, Sequence[Integral], numpy.ndarray]
        The radial order(s) :math:`n`.

    m : Union[Integral, Sequence[Integral], numpy.ndarray]
        The azimuthal frequency(ies) :math:`m`.

    parameter : Union[str, Sequence[str], numpy.ndarray]
        The parameter(s) with respect to which the polynomials are derived, among ``"x0"``, ``"y0"``,
        ``"Rx"``, ``"Ry"``, ``"alpha"`` and ``"h"``.

    Rx : Optional[Real], optional
        The semi-axis :math:`R_x` of the ellipse along its x axis. Must be finite and strictly positive. Default is 1.0.

    Ry : Optional[Real], optional
        The semi-axis :math:`R_y` of the ellipse along its y axis. Must be finite and strictly positive. Default is 1.0.

    x0 : Optional[Real], optional
        The x coordinate :math:`x_0` of the center of the ellipse. Must be finite. Default is 0.0.

    y0 : Optional[Real], optional
        The y coordinate :math:`y_0` of the center of the ellipse. Must be finite. Default is 0.0.

    alpha : Optional[Real], optional
        The rotation angle :math:`\alpha` of the ellipse in radians. Must be finite. Default is 0.0.

    h : Optional[Real], optional
        The ratio :math:`h` of the inner to the outer semi-axes, in :math:`[0, 1)`. Default is 0.0 (full ellipse).

    default : Union[int, float], optional
        The value set outside the elliptic annulus or where the coordinates are not finite. Integers and floating
        point numbers (Python or numpy, including ``numpy.nan`` and ``numpy.inf``) are accepted and converted to
        float. Booleans are not accepted. Default is ``numpy.nan``.

    precompute : bool, optional
        If True, the powers of :math:`\rho_{eq}` and the cosine and sine terms are cached and shared between the
        polynomials, which is faster when several derivatives are computed at once. If False, they are
        recomputed for each polynomial, which uses less memory. Default is True.

    normalize : bool, optional
        If True, the derivatives are multiplied by the normalization factors of Noll
        :math:`N_{n}^{m} = \sqrt{(2 - \delta_{m0})(n + 1)}` (see :func:`pyzernike.zernike_normalization`).
        The value ``default`` is not normalized. Default is False.

    to_list : bool, optional
        Only used when several derivatives are computed. If True, the output is a list of arrays.
        If False, the output is a single array of shape :math:`(N, \dots)`. Default is True.

    None is accepted for all the parameters of the ellipse, and means the default value.

    Returns
    -------
    Union[numpy.ndarray, List[numpy.ndarray]]
        - If ``n``, ``m`` and ``parameter`` are single values: an array with the shape of ``x``.
        - Otherwise, if ``to_list`` is True: a list of :math:`N` arrays with the shape of ``x``.
        - Otherwise: an array of shape :math:`(N, \dots)`, where ``output[i]`` is the derivative of the
          polynomial ``(n[i], m[i])`` with respect to ``parameter[i]``.

        The dtype is the common floating point dtype of ``x`` and ``y`` (``numpy.float64`` by default).

    Raises
    ------
    TypeError
        If ``x`` or ``y`` can not be converted to a floating point array, if a parameter of the ellipse
        is not a real number, if ``default`` is not an integer or a floating point number, if ``n`` or ``m``
        is not an integer or a sequence of integers, if ``parameter`` is not a string or a sequence of
        strings, or if ``precompute``, ``normalize`` or ``to_list`` is not a boolean.

    ValueError
        If ``x`` and ``y`` do not have the same shape, if a parameter of the ellipse is not finite, if
        ``Rx`` or ``Ry`` is not strictly positive, if ``h`` is not in :math:`[0, 1)`, if ``parameter``
        contains an unknown name, or if the sequences do not have the same length.

    References
    ----------
    .. [1] R. Navarro, J. L. López, J. A. Díaz and E. Pérez Sinusía, "Generalization of Zernike polynomials for
       regular portions of circles and ellipses," Optics Express 22(18), 21263-21279 (2014).
       https://doi.org/10.1364/OE.22.021263

    Examples
    --------
    Compute the derivative of the defocus :math:`Z_{2}^{0}` with respect to the center :math:`x_0` of a
    circle of radius 10 centered at :math:`(1, 1)`:

    .. code-block:: python

        import numpy
        from pyzernike import xy_zernike_parameter_derivative

        X, Y = numpy.meshgrid(numpy.linspace(-10, 10, 100), numpy.linspace(-10, 10, 100))
        dZ_dx0 = xy_zernike_parameter_derivative(X, Y, n=2, m=0, parameter="x0", Rx=10, Ry=10, x0=1.0, y0=1.0)

    Compute the gradient of :math:`Z_{2}^{0}` with respect to all the parameters (``n`` and ``m`` are repeated):

    .. code-block:: python

        parameters = ["x0", "y0", "Rx", "Ry", "alpha", "h"]
        gradient = xy_zernike_parameter_derivative(X, Y, n=2, m=0, parameter=parameters, Rx=10, Ry=10, x0=1.0, y0=1.0, to_list=False)
        # array of shape (6, 100, 100)

    """
    x, y = _as_common_floating_arrays(x=x, y=y)
    Rx, Ry, x0, y0, alpha, h = _as_ellipse_parameters(Rx, Ry, x0, y0, alpha, h)
    default = _as_default_value(default)
    precompute = _check_boolean(precompute, "precompute")
    normalize = _check_boolean(normalize, "normalize")
    to_list = _check_boolean(to_list, "to_list")
    indices, single = _broadcast_parsed({
        "n": _as_integers(n, "n"),
        "m": _as_integers(m, "m"),
        "parameter": _as_choices(parameter, "parameter", ELLIPSE_PARAMETERS),
    })

    float_type = x.dtype.type
    rho_list, theta_list = _core_cartesian_to_elliptic_annulus(
        x=x,
        y=y,
        Rx=float_type(Rx),
        Ry=float_type(Ry),
        x0=float_type(x0),
        y0=float_type(y0),
        alpha=float_type(alpha),
        h=float_type(h),
        x_derivative=[0],
        y_derivative=[0],
    )
    rho_eq, theta_eq = rho_list[0], theta_list[0]

    valid_mask = numpy.isfinite(rho_eq) & numpy.isfinite(theta_eq) & (rho_eq >= 0) & (rho_eq <= 1)

    output = numpy.full((len(indices["n"]),) + x.shape, default, dtype=x.dtype)
    output[:, valid_mask] = _core_xy_parameter_derivatives(
        rho=rho_eq[valid_mask],
        theta=theta_eq[valid_mask],
        n=indices["n"],
        m=indices["m"],
        parameter=indices["parameter"],
        Rx=Rx,
        Ry=Ry,
        alpha=alpha,
        h=h,
        precompute=precompute,
        normalize=normalize,
    )

    return _format_output(output, single, to_list)