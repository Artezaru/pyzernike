import numpy
from numbers import Integral, Real
from typing import Sequence, List, Optional
from scipy.special import gammaln

from .zernike_polynomial_up_to_order import zernike_polynomial_up_to_order

def xy_zernike_polynomial_up_to_order(
    x: numpy.ndarray,
    y: numpy.ndarray,
    order: Integral,
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
) -> List[List[numpy.ndarray]]:
    r"""
    This method computes all the Zernike polynomials in an extended domain G up to a given order for different orders and degrees, including their derivatives with respect to the x and y coordinates.

    ``output[k][j]`` is the Zernike polynomial of order ``n[j]`` and degree ``m[j]`` (OSA/ANSI ordering) with the x derivative of order ``x_derivative[k]`` and the y derivative of order ``y_derivative[k]``.

    This method is more optimized than the :func:`pyzernike.core_polynomial` method, as it assemblate the polynomials boucling on the radial parts and not the orders and degrees to avoid recomputing the same radial parts multiple times.

    if :math:`\rho` is not in :math:`0 \leq \rho \leq 1` or :math:`\rho` is numpy.nan, the output is set to the default value (numpy.nan by default).

    .. seealso::

        - :func:`pyzernike.xy_zernike_polynomial` for computing selected Zernike polynomial :math:`Z_{n}^{m}(x, y)`.
        - :func:`pyzernike.zernike_index_to_order` to extract the Zernike orders (n, m) from the indices.

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
    
    The :math:`\rho` and :math:`\theta` values are the same for all the polynomials, and the orders and degrees are provided as sequences. 
    Moreover, the domain is the same for all the polynomials.
    
    .. note::

        The alias ``Zxyup`` is available for this function.

    .. note::

        For developers, the ``_skip`` parameter is used to skip the checks for the input parameters. This is useful for internal use where the checks are already done.
        In this case :

        - ``x`` and ``y`` must be numpy.ndarray in the valid domain with finite values and float64 dtype and the same shape.
        - ``x_derivative`` and ``y_derivative`` must be given as sequence of integers with the same length and valid values.

        See also :func:`pyzernike.core_polynomial` for more details on the input parameters.

    .. warning::

        The derivatives with respect to x and y are implemented for (x_derivative, y_derivative) = (0, 0), (1, 0), (0, 1) only.
        Faa di Bruno's formula can be used to compute higher order derivatives, but it is not implemented in this function.

    The derivatives with respect to x and y are computed using the chain rule.

    .. math::

        \frac{\partial Zxy_{n}^{m}}{\partial z} = \frac{\partial Z_{n}^{m}}{\partial \rho_{eq}} \cdot \frac{\partial \rho_{eq}}{\partial z} + \frac{\partial Z_{n}^{m}}{\partial \theta_{eq}} \cdot \frac{\partial \theta_{eq}}{\partial z}

    where:

    .. math::

        \frac{\partial \rho_{eq}}{\partial z} = \frac{1}{1 - h} \cdot \frac{1}{r} \cdot \left( \frac{X}{R_x^2} \cdot \frac{\partial X}{\partial z} + \frac{Y}{R_y^2} \cdot \frac{\partial Y}{\partial z} \right)

    .. math::

        \frac{\partial \theta_{eq}}{\partial z} = \frac{2 \pi}{\theta_2 - \theta_1} \cdot \frac{1}{R_x R_y r^2} \cdot \left( X \cdot \frac{\partial Y}{\partial z} - Y \cdot \frac{\partial X}{\partial z} \right)

    Parameters
    ----------
    x : numpy.ndarray (N-D array)
        The cartesian x values with shape (...,).

    y : numpy.ndarray (N-D array)
        The cartesian y values with shape (...,). Same shape as x.

    order : Integral
        The maximum order of the Zernike polynomials to compute. The orders and degrees are given as sequences of integers.

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
    List[List[numpy.ndarray]]
        A list of lists containing the Zernike polynomials evaluated at the points (x, y) for each order and degree.
        ``output[k][j]`` is the Zernike polynomial of order ``n[j]`` and degree ``m[j]`` (OSA/ANSI ordering) with the x derivative of order ``x_derivative[k]`` and the y derivative of order ``y_derivative[k]``.

    Raises
    ------
    TypeError
        If `x` or `y` are not numpy arrays, or if `order` is not an integer, or if `x_derivative` or `y_derivative` are not sequences of integers.

    ValueError
        If `x` and `y` do not have the same shape, or if `order` is negative, or if `x_derivative` or `y_derivative` are not of the same length, or if `x_derivative` or `y_derivative` contain negative integers.

    Examples
    --------
    
    Compute all the Zernike polynomials up to order 3 for a cartesian grid of points in the domain defined by a circle with radius sqrt(2) centered at (0, 0):

    .. code-block:: python

        import numpy
        from pyzernike import xy_zernike_polynomial_up_to_order, zernike_index_to_order

        # Create a grid of points
        x = numpy.linspace(-1, 1, 100)
        y = numpy.linspace(-1, 1, 100)
        x, y = numpy.meshgrid(x, y)  # Create a 2D grid

        # Compute the Zernike polynomials up to order 3
        result = xy_zernike_polynomial_up_to_order(x, y, order=3, Rx=numpy.sqrt(2), Ry=numpy.sqrt(2), x0=0, y0=0)
        polynomials = result[0]  # Get the first set of polynomials (for x_derivative=0, y_derivative=0)

        # Extract the values: 
        indices = list(range(len(polynomials)))
        n, m = zernike_index_to_order(indices)  # Get the orders and degrees from the indices

        for i, (n_i, m_i) in enumerate(zip(n, m)):
            print(f"Zernike polynomial Z_{n_i}^{m_i} for the given x and y values is: {polynomials[i]}")

    To compute the polynomials and their first derivatives with respect to x:

    .. code-block:: python

        import numpy
        from pyzernike import xy_zernike_polynomial_up_to_order, zernike_index_to_order

        # Create a grid of points
        x = numpy.linspace(-1, 1, 100)
        y = numpy.linspace(-1, 1, 100)
        x, y = numpy.meshgrid(x, y)  # Create a 2D grid

        # Compute the Zernike polynomials up to order 3 with x derivatives
        result = xy_zernike_polynomial_up_to_order(x, y, order=3, x_derivative=[0, 1], Rx=numpy.sqrt(2), Ry=numpy.sqrt(2), x0=0, y0=0)
        polynomials = result[0]  # Get the first set of polynomials (for x_derivative=0, y_derivative=0)
        derivatives_x = result[1]  # Get the first set of derivatives (for x_derivative=1, y_derivative=0)

    The output will contain the Zernike polynomials and their derivatives for the specified orders and degrees.
    """
    if not _skip:
        x = numpy.asarray(x, dtype=numpy.float64)
        y = numpy.asarray(y, dtype=numpy.float64)
        if not isinstance(order, Integral) or order < 0:
            raise TypeError("Order must be a non-negative integer.")
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
        if x_derivative is not None and y_derivative is not None and len(x_derivative) != len(y_derivative):
            raise ValueError("x_derivative and y_derivative must have the same length.")
        if y_derivative is not None and x_derivative is None:
            x_derivative = [0] * len(y_derivative)
        if x_derivative is not None and y_derivative is None:
            y_derivative = [0] * len(x_derivative)
        if x_derivative is None and y_derivative is None:
            x_derivative = [0]
            y_derivative = [0]

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
        
        for index in range(len(x_derivative)):
            if (x_derivative[index], y_derivative[index]) not in [(0, 0), (1, 0), (0, 1)]:
                raise ValueError("The function supports only the derivatives (0, 0), (1, 0) and (0, 1). For more complex derivatives, use the standard Zernike polynomial function with polar coordinates.")

    # Create the output list
    max_index = (order * (order + 3)) // 2
    output = [[None for _ in range(max_index + 1)] for _ in range(len(x_derivative))]

    # =================================================================
    # Convert the x and y coordinates to polar coordinates in the extended domain G
    # =================================================================   
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
    r = numpy.sqrt((X / Rx) ** 2 + (Y / Ry) ** 2)
    theta = numpy.arctan2(Y / Ry, X / Rx)

    # Angular convertion in 0 to 2*pi range
    theta_prim_2pi = theta % (2 * numpy.pi)
    theta1_2pi = theta1 % (2 * numpy.pi)
    theta2_2pi = theta2 % (2 * numpy.pi)

    # Compute the equivalent rho values
    rho_eq = (r - h) / (1 - h)
    
    # Compute the equivalent theta values
    if closed_circle:
        # theta_1 = theta_2 , theta = 0 for theta_1
        t = theta_prim_2pi
        t1 = t2 = theta1_2pi
        theta_eq = t - t1
    elif theta1_2pi < theta2_2pi and not closed_circle:
        # 0 -------[t1, t2]------- 2*pi
        t = theta_prim_2pi
        t1 = theta1_2pi
        t2 = theta2_2pi
        theta_eq = 2 * numpy.pi * (t - t1) / (t2 - t1)
    elif theta1_2pi > theta2_2pi and not closed_circle:
        # [0, t2]-------[t1, 2*pi]
        # Transform theta_prim_2pi to be in [theta1, theta1 + 2 * pi]
        t = theta_prim_2pi.copy()
        t[t < theta1_2pi] += 2 * numpy.pi # Define know in [theta1, theta1 + 2 * pi]
        t1 = theta1_2pi
        t2 = theta2_2pi + 2 * numpy.pi
        theta_eq = 2 * numpy.pi * (t - t1) / (t2 - t1)
    else:
        raise ValueError("Invalid theta1 and theta2 values. They must define a valid sector.")
    
    # =================================================================
    # Compute the orders and degrees of the Zernike polynomials for the rho and theta values
    # =================================================================
    # If at least one derivative is needed, we compute also the derivative along x and y
    compute_derivative = False
    comptute_polynomial = False
    
    if any([(x_derivative[index], y_derivative[index]) in [(0, 0)] for index in range(len(x_derivative))]):
        comptute_polynomial = True
    if any([(x_derivative[index], y_derivative[index]) in [(1, 0), (0, 1)] for index in range(len(x_derivative))]):
        compute_derivative = True
    
    # Create the list of derivatives and orders to compute
    rho_derivative = []
    theta_derivative = []
    if comptute_polynomial:
        rho_derivative.extend([0])
        theta_derivative.extend([0])
    if compute_derivative:
        rho_derivative.extend([1, 0])  # rho derivative
        theta_derivative.extend([0, 1])  # theta derivative
    
    # Compute the polynomials using the zernike_polynomial_up_to_order function
    zernike_polynomials = zernike_polynomial_up_to_order(
        rho=rho_eq,
        theta=theta_eq,
        order=order,
        rho_derivative=rho_derivative,
        theta_derivative=theta_derivative,
        _skip=_skip
    )

    # We want, index=0 -> polynomials, index=1 -> rho derivatives, index=2 -> theta derivatives
    # To ensure that, add a None at the beginning of the output list if compute_polynomial is False to avoid index shift
    if not comptute_polynomial:
        zernike_polynomials.insert(0, None)

    # =================================================================
    # Precompute the radial and angular derivatives if needed
    # =================================================================
    # Construct the output arrays for the derivatives
    if compute_derivative:
        rho_derivative_x = (1 / (1 - h)) * (1 / r) * ( (X / (Rx**2)) * numpy.cos(alpha) - (Y / (Ry**2)) * numpy.sin(alpha))
        theta_derivative_x = (2 * numpy.pi / (theta2 - theta1)) * (1 / (Rx * Ry * r**2)) * ( - X * numpy.sin(alpha) - Y * numpy.cos(alpha))
        rho_derivative_y = (1 / (1 - h)) * (1 / r) * ( (X / (Rx**2)) * numpy.sin(alpha) + (Y / (Ry**2)) * numpy.cos(alpha))
        theta_derivative_y = (2 * numpy.pi / (theta2 - theta1)) * (1 / (Rx * Ry * r**2)) * ( X * numpy.cos(alpha) - Y * numpy.sin(alpha))
        
    for n in range(order + 1):
        # We boucle only over the positive m values, as the negative m values are computed by symmetry.
        for m in range(-n, n + 1, 2): # positive and negative m values
            zernike_index = (n * (n + 2) + m) // 2
            for derivative_index in range(len(x_derivative)):
                x_derivative_idx = x_derivative[derivative_index]
                y_derivative_idx = y_derivative[derivative_index]

                if x_derivative_idx == 1 and y_derivative_idx == 0:
                    output[derivative_index][zernike_index] = zernike_polynomials[1][zernike_index] * rho_derivative_x + zernike_polynomials[2][zernike_index] * theta_derivative_x
                elif x_derivative_idx == 0 and y_derivative_idx == 1:
                    output[derivative_index][zernike_index] = zernike_polynomials[1][zernike_index] * rho_derivative_y + zernike_polynomials[2][zernike_index] * theta_derivative_y
                elif x_derivative_idx == 0 and y_derivative_idx == 0:
                    output[derivative_index][zernike_index] = zernike_polynomials[0][zernike_index]

    return output
        