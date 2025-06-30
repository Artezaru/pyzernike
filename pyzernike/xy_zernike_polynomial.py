import numpy
from numbers import Integral, Real
from typing import Sequence, List, Optional

from .core_polynomial import core_polynomial

def xy_zernike_polynomial(
    x: numpy.ndarray,
    y: numpy.ndarray,
    n: Sequence[Integral],
    m: Sequence[Integral],
    x_derivative: Optional[Sequence[Integral]] = None,
    y_derivative: Optional[Sequence[Integral]] = None,
    default: Real = numpy.nan,
    Rx: float = 1.0, 
    Ry: float = 1.0, 
    x0: float = 0.0, 
    y0: float = 0.0, 
    alpha: float = 0.0, 
    h: float = 0.0, 
    theta1: float = 0.0, 
    theta2: float = 2 * numpy.pi,
    _skip: bool = False
) -> List[numpy.ndarray]:
    r"""
    Computes the Zernike polynomial :math:`Z_{n}^{m}(\rho, \theta)` for given cartesian coordinates :math:`(x, y)`.

    If :math:`|m| > n` or :math:`n < 0`, or :math:`(n - m)` is odd, the output is a zeros array with the same shape as :math:`\rho`.

    if :math:`\rho` is not in :math:`0 \leq \rho \leq 1` or :math:`\rho` is numpy.nan, the output is set to the default value (numpy.nan by default).

    .. seealso::

        :func:`pyzernike.zernike_polynomial` for computing the radial Zernike polynomial :math:`Z_{n}^{m}` for polar coordinates :math:`(\rho, \theta)`. 

    .. seealso::

        For the mathematical development of the method, see the paper `Generalization of Zernike polynomials for regular portions of circles and ellipses` by Rafael Navarro, José L. López, José Rx. Díaz, and Ester Pérez Sinusía.
        The associated paper is available in the resources folder of the package.

        Download the PDF : :download:`PDF <../../../pyzernike/resources/Navarro and al. Generalization of Zernike polynomials for regular portions of circles and ellipses.pdf>`

    The user must provide the x and y coordinates of the points where the polynomial is evaluated and the parameters of the extended domain G:

    - :math:`R_x` and :math:`R_y` are the lenght of the semi-axis of the ellipse (outer boundary noted A and B on the figure).
    - :math:`x_0` and :math:`y_0` are the coordinates of the center of the ellipse.
    - :math:`\alpha` is the rotation angle of the ellipse in radians.
    - :math:`h=\frac{a}{R_x}=\frac{b}{R_y}` defining the inner boundary of the ellipse.
    - :math:`\theta_1` and :math:`\theta_2` are the angles defining the sector of the ellipse where the polynomial is described.

    .. figure:: ../../../pyzernike/resources/extended_parameters.png
        :width: 400px
        :align: center

        The parameters to define the extended domain of the Zernike polynomial. ``A`` and ``B`` are the semi-major and semi-minor axes of the ellipse, respectively, they correspond to :math:`R_x` and :math:`R_y`.

    The applied mapping is as follows:

    .. math::

        Zxy_{n}^{m}(x, y) = Z_{n}^{m}\left(\frac{r - h}{1 - h}, \frac{2 \pi (\theta - \theta_1)}{\theta_2 - \theta_1}\right)

    Where:

    .. math::

        r = \sqrt{\left(\frac{X}{R_x}\right)^{2} + \left(\frac{Y}{R_y}\right)^{2}}

    .. math::

        \theta = \text{atan2} (\frac{Y}{R_y}, \frac{X}{R_x})

    .. math::

        X = \cos(\alpha) (x - x_0) + \sin(\alpha) (y - y_0)
    
    .. math::

        Y = -\sin(\alpha) (x - x_0) + \cos(\alpha) (y - y_0)
    
    This function allows to compute several Zernike polynomials at once for different orders and degrees, which can be more efficient than calling the radial polynomial function multiple times.
    The :math:`\rho` and :math:`\theta` values are the same for all the polynomials, and the orders and degrees are provided as sequences. 
    Moreover, the domain is the same for all the polynomials.
    The output is a list of numpy arrays, each containing the values of the Zernike polynomial for the corresponding order and degree.
    
    .. note::

        The alias ``Zxy`` is available for this function.

    .. note::

        For developers, the ``_skip`` parameter is used to skip the checks for the input parameters. This is useful for internal use where the checks are already done.
        In this case :

        - ``x`` and ``y`` must be numpy.ndarray in the valid domain with finite values and float64 dtype and the same shape.
        - ``n``, ``m``, ``x_derivative`` and ``y_derivative`` must be given as sequence of integers with the same length and valid values.

        See also :func:`pyzernike.core_polynomial` for more details on the input parameters.

    .. warning::

        The derivatives with respect to x and y are implemented for (x_derivative, y_derivative) = (0, 0), (1, 0), (0, 1) only.
        Faa di Bruno's formula can be used to compute higher order derivatives, but it is not implemented in this function.

    Parameters
    ----------
    x : numpy.ndarray (N-D array)
        The cartesian x values with shape (...,).

    y : numpy.ndarray (N-D array)
        The cartesian y values with shape (...,). Same shape as x.

    n : Union[Integral, Sequence[Integral]]
        A list of the radial order(s) of the Zernike polynomial(s) to compute.

    m : Union[Integral, Sequence[Integral]]
        A list of the radial degree(s) of the Zernike polynomial(s) to compute.

    x_derivative : Optional[Union[Integral, Sequence[Integral]]], optional
        A list of the order(s) of the x derivative(s) to compute, expected to have shape=(Npolynomials,).
        If None, is it assumed that x_derivative is 0 for all polynomials.

    y_derivative : Optional[Union[Integral, Sequence[Integral]]], optional
        A list of the order(s) of the angular derivative(s) to compute, expected to have shape=(Npolynomials,).
        If None, is it assumed that y_derivative is 0 for all polynomials.

    default : Real, optional
        The default value for invalid rho values. The default is numpy.nan.
        If the radial coordinate values are not in the valid domain (0 <= rho <= 1) or if they are numpy.nan, the output is set to this value.

    Rx : float, optional
        The length of the semi-major axis of the ellipse (outer boundary). The default is 1.0. Must be greater than 0.

    Ry : float, optional
        The length of the semi-minor axis of the ellipse (outer boundary). The default is 1.0. Must be greater than 0.
    
    x0 : float, optional
        The x-coordinate of the center of the ellipse. The default is 0.0.
    
    y0 : float, optional
        The y-coordinate of the center of the ellipse. The default is 0.0.

    alpha : float, optional
        The rotation angle of the ellipse in radians. The default is 0.0.

    h : float, optional
        The ratio of the inner semi-axis to the outer semi-axis. The default is 0. Must be in the range [0, 1[.
    
    theta1 : float, optional
        The starting angle of the sector in radians. The default is 0.0.

    theta2 : float, optional
        The ending angle of the sector in radians. The default is 2 * pi. Must be greater than theta1 and less than or equal to theta1 + 2 * pi.

    _skip : bool, optional
        If True, the checks for the input parameters are skipped. This is useful for internal use where the checks are already done.
        The default is False.

    Returns
    -------
    List[numpy.ndarray]
        A list of numpy arrays containing the Zernike polynomial values for each order and degree.
        Each array has the same shape as ``rho``.

    Raises
    ------
    TypeError
        If the rho or theta values are not a numpy array or if n and m are not integers.

    ValueError
        If the lengths of n and m are not the same.
        If the shape of rho and theta are not the same.

    Examples
    --------
    Let's consider a full circle with a radius of 10 centered at the origin (0, 0).
    The value of the zernike polynomial :math:`Z_{2}^{0}` at the point (x, y) is given by:

    .. code-block:: python

        import numpy
        from pyzernike import xy_zernike_polynomial # or Zxy
        x = numpy.linspace(-10, 10, 100)
        y = numpy.linspace(-10, 10, 100)
        X, Y = numpy.meshgrid(x, y)

        zernike = xy_zernike_polynomial(X, Y, n=[2], m=[0], Rx=10, Ry=10, x0=0.0, y0=0.0) # Shape similar to X and Y

    returns a list with a single numpy array containing the values of the Zernike polynomial :math:`Z_{2}^{0}` at the points (X, Y) within the circle of radius 10.
    """
    if not _skip:
        x = numpy.asarray(x, dtype=numpy.float64)
        y = numpy.asarray(y, dtype=numpy.float64)
        if not isinstance(n, Sequence) or not all(isinstance(i, Integral) for i in n):
            raise TypeError("n must be a sequence of integers.")
        if not isinstance(m, Sequence) or not all(isinstance(i, Integral) for i in m):
            raise TypeError("m must be a sequence of integers.")
        if x_derivative is not None:
            if not isinstance(x_derivative, Sequence) or not all(isinstance(i, Integral) and i >= 0 for i in x_derivative):
                raise TypeError("x_derivative must be a sequence of non-negative integers.")
        if y_derivative is not None:
            if not isinstance(y_derivative, Sequence) or not all(isinstance(i, Integral) and i >= 0 for i in y_derivative):
                raise TypeError("y_derivative must be a sequence of non-negative integers.")
        if not isinstance(default, Real):
            raise TypeError("Default value must be a real number.")
        
        if not x.shape == y.shape:
            raise ValueError("x and y must have the same shape.")
        if len(n) != len(m):
            raise ValueError("n and m must have the same length.")
        if x_derivative is not None and len(n) != len(x_derivative):
            raise ValueError("n and x_derivative must have the same length.")
        if y_derivative is not None and len(n) != len(y_derivative):
            raise ValueError("n and y_derivative must have the same length.")
        if x_derivative is None:
            x_derivative = [0] * len(n)
        if y_derivative is None:
            y_derivative = [0] * len(n)

        if not isinstance(Rx, Real) or Rx <= 0:
            raise TypeError("Rx must be a positive real number.")
        if not isinstance(Ry, Real) or Ry <= 0:
            raise TypeError("Ry must be a positive real number.")
        if not isinstance(x0, Real):
            raise TypeError("x0 must be a real number.")
        if not isinstance(y0, Real):
            raise TypeError("y0 must be a real number.")
        if not isinstance(alpha, Real):
            raise TypeError("Alpha must be a real number.")
        if not isinstance(h, Real) or not (0 <= h < 1):
            raise TypeError("h must be a real number in the range [0, 1[.")
        if not isinstance(theta1, Real):
            raise TypeError("Theta1 must be a real number.")
        if not isinstance(theta2, Real):
            raise TypeError("Theta2 must be a real number.")
        if abs(theta2 - theta1) > 2 * numpy.pi:
            raise ValueError("The angle between theta1 and theta2 must be less than or equal to 2 * pi.")
        if theta1 >= theta2:
            raise ValueError("Theta1 must be less than Theta2.")
        
        for index in range(len(n)):
            if (x_derivative[index], y_derivative[index]) not in [(0, 0), (1, 0), (0, 1)]:
                raise ValueError("The function supports only the derivatives (0, 0), (1, 0) and (0, 1). For more complex derivatives, use the standard Zernike polynomial function with polar coordinates.")
        
    # Complete circle case
    if abs(theta2 - theta1 - 2 * numpy.pi) < 1e-10:
        closed_circle = True
    else:
        closed_circle = False

    # Computing the X, Y arrays from x and y coordinates
    x_centered = x - x0
    y_centered = y - y0
    X = numpy.cos(alpha) * x_centered + numpy.sin(alpha) * y_centered
    Y = - numpy.sin(alpha) * x_centered + numpy.cos(alpha) * y_centered

    # Compute the equivalent polar coordinates
    r_prim = numpy.sqrt((X / Rx) ** 2 + (Y / Ry) ** 2)
    theta_prim = numpy.arctan2(Y / Ry, X / Rx)

    # Angular convertion in 0 to 2*pi range
    theta_prim_2pi = theta_prim % (2 * numpy.pi)
    theta1_2pi = theta1 % (2 * numpy.pi)
    theta2_2pi = theta2 % (2 * numpy.pi)

    # Compute the equivalent rho values
    rho = (r_prim - h) / (1 - h)
    
    # Compute the equivalent theta values
    if closed_circle:
        # theta_1 = theta_2 , theta = 0 for theta_1
        t = theta_prim_2pi
        t1 = t2 = theta1_2pi
        theta = t - t1
    elif theta1_2pi < theta2_2pi and not closed_circle:
        # 0 -------[t1, t2]------- 2*pi
        t = theta_prim_2pi
        t1 = theta1_2pi
        t2 = theta2_2pi
        theta = 2 * numpy.pi * (t - t1) / (t2 - t1)
    elif theta1_2pi > theta2_2pi and not closed_circle:
        # [0, t2]-------[t1, 2*pi]
        # Transform theta_prim_2pi to be in [theta1, theta1 + 2 * pi]
        t = theta_prim_2pi.copy()
        t[t < theta1_2pi] += 2 * numpy.pi # Define know in [theta1, theta1 + 2 * pi]
        t1 = theta1_2pi
        t2 = theta2_2pi + 2 * numpy.pi
        theta = 2 * numpy.pi * (t - t1) / (t2 - t1)
    else:
        raise ValueError("Invalid theta1 and theta2 values. They must define a valid sector.")
    


    # Check the validity of the radial coordinate values
    if not _skip:
        # Compute the Mask for valid rho values
        domain_mask = (rho >= 0) & (rho <= 1)
        finite_mask = numpy.isfinite(rho) & numpy.isfinite(theta)
        valid_mask = domain_mask & finite_mask

        # Conserve only the valid values and save the input shape
        original_shape = rho.shape
        rho = rho[valid_mask]
        theta = theta[valid_mask]


    # If derivatives are not 0 -> rho_derivative and theta_derivative are 1 to compute the first derivative
    z_n = []
    z_m = []
    rho_derivative = []
    theta_derivative = []
    compute_derivative = False
    for index in range(len(n)):
        if x_derivative[index] == 1 or y_derivative[index] == 1:
            # If the derivative is 1, we compute the first derivative
            z_n.extend([n[index], n[index]])
            z_m.extend([m[index], m[index]])
            rho_derivative.extend([1, 0])
            theta_derivative.extend([0, 1])
            compute_derivative = True
        else:
            # If the derivative is 0, we compute the polynomial
            z_n.append(n[index])
            z_m.append(m[index])
            rho_derivative.append(0)
            theta_derivative.append(0)

    # Compute the polynomials using the core_polynomial function
    zernike_polynomials = core_polynomial(
        rho=rho,
        theta=theta,
        n=z_n,
        m=z_m,
        rho_derivative=rho_derivative,
        theta_derivative=theta_derivative,
        flag_radial=False
    )

    # If rho is not in the valid domain, set the output to the default value
    if not _skip:
        for index in range(len(zernike_polynomials)):
            # Reshape the radial polynomial to the original shape of rho and set the invalid values to the default value
            output_default = numpy.full(original_shape, default, dtype=numpy.float64)
            output_default[valid_mask] = zernike_polynomials[index]
            zernike_polynomials[index] = output_default

    # Construct the output arrays for the derivatives
    if compute_derivative:
        rho_derivative_x = (1 / (1 - h)) * (1 / r_prim) * ( (X / Rx**2) * numpy.cos(alpha) - (Y / Ry**2) * numpy.sin(alpha))
        theta_derivative_x = (2 * numpy.pi / (theta2 - theta1)) * (1 / (Rx * Ry * r_prim**2)) * ( - X * numpy.sin(alpha) - Y * numpy.cos(alpha))
        rho_derivative_y = (1 / (1 - h)) * (1 / r_prim) * ( (X / Rx**2) * numpy.sin(alpha) + (Y / Ry**2) * numpy.cos(alpha))
        theta_derivative_y = (2 * numpy.pi / (theta2 - theta1)) * (1 / (Rx * Ry * r_prim**2)) * ( X * numpy.cos(alpha) - Y * numpy.sin(alpha))
        
    output = []
    zernike_index = 0
    for index in range(len(n)):
        if x_derivative[index] == 1:
            output.append(zernike_polynomials[zernike_index] * rho_derivative_x + zernike_polynomials[zernike_index + 1] * theta_derivative_x)
            zernike_index += 2
        elif y_derivative[index] == 1:
            output.append(zernike_polynomials[zernike_index] * rho_derivative_y + zernike_polynomials[zernike_index + 1] * theta_derivative_y)
            zernike_index += 2
        else:
            output.append(zernike_polynomials[zernike_index])
            zernike_index += 1
    
    return output
        