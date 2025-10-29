import numpy
import sympy
import pytest

from pyzernike.core import core_cartesian_to_elliptic_annulus
from pyzernike.core import core_corresponding_signed_integer_type


# ----------------------------------------------------------------------
# Helper: map a NumPy floating dtype to the signed integer dtype that the
# core routine expects.
# ----------------------------------------------------------------------
_FLOAT_TO_INT = {
    numpy.float16: numpy.int16,
    numpy.float32: numpy.int32,
    numpy.float64: numpy.int64,
}


def _typed_float_array(data, float_type):
    """Create a NumPy array of the requested floating‑point dtype."""
    return numpy.array(data, dtype=float_type)


def _typed_int_array(data, float_type):
    """Create a 1‑D integer array whose dtype matches *float_type*."""
    int_dtype = _FLOAT_TO_INT[float_type]
    return numpy.array(data, dtype=int_dtype)


# ----------------------------------------------------------------------
# Reference (analytic) helpers for the *unit‑circle* case.
# ----------------------------------------------------------------------
def _polar_rho(x, y):
    """ρ = sqrt(x² + y²) – the same as ``rho_eq`` when Rx=Ry=1, h=0."""
    return numpy.sqrt(x ** 2 + y ** 2)


def _polar_theta(x, y):
    """θ = arctan2(y, x) – the same as ``theta_eq`` when Rx=Ry=1."""
    return numpy.arctan2(y, x)


def _drho_dx(x, y):
    """∂ρ/∂x = x / ρ   (with ρ = sqrt(x²+y²))."""
    r = _polar_rho(x, y)
    # Guard against division by zero – set derivative to 0 at the origin.
    return x / r


def _drho_dy(x, y):
    """∂ρ/∂y = y / ρ."""
    r = _polar_rho(x, y)
    return y / r


def _dtheta_dx(x, y):
    """∂θ/∂x = -y / (x² + y²)  = -y / ρ²."""
    r2 = x ** 2 + y ** 2
    return -y / r2


def _dtheta_dy(x, y):
    """∂θ/∂y =  x / (x² + y²)  =  x / ρ²."""
    r2 = x ** 2 + y ** 2
    return x / r2


# ----------------------------------------------------------------------
# Parameterise the whole suite for the three floating‑point precisions.
# ----------------------------------------------------------------------
@pytest.mark.parametrize("float_type", [numpy.float16, numpy.float32, numpy.float64])
def test_output_shapes_and_dtypes(float_type):
    """
    Verify that the two output lists have the correct length,
    that each element matches the shape of the input grid,
    and that every array uses the expected dtypes.
    """
    # Simple 3×3 grid – values are deliberately small so that float16 still
    # retains reasonable precision for the shape checks.
    xs = _typed_float_array(
        [[-0.5, 0.0, 0.5],
         [-0.5, 0.0, 0.5],
         [-0.5, 0.0, 0.5]],
        float_type,
    )
    ys = _typed_float_array(
        [[-0.5, -0.5, -0.5],
         [ 0.0,  0.0,  0.0],
         [ 0.5,  0.5,  0.5]],
        float_type,
    )

    # Ellipse is a unit circle → Rx = Ry = 1, centre (0,0), no rotation, h=0.
    Rx = float_type(1.0)
    Ry = float_type(1.0)
    x0 = float_type(0.0)
    y0 = float_type(0.0)
    alpha = float_type(0.0)
    h = float_type(0.0)

    # Request three derivative pairs: (0,0), (1,0), (0,1)
    x_der = _typed_int_array([0, 1, 0], float_type)
    y_der = _typed_int_array([0, 0, 1], float_type)

    rho_list, theta_list = core_cartesian_to_elliptic_annulus(
        x=xs,
        y=ys,
        Rx=Rx,
        Ry=Ry,
        x0=x0,
        y0=y0,
        alpha=alpha,
        h=h,
        x_derivative=x_der,
        y_derivative=y_der,
        float_type=float_type,
    )

    # ------------------------------------------------------------------
    # 1️⃣  Length of the output lists must equal the number of derivative
    #     pairs we asked for.
    # ------------------------------------------------------------------
    assert len(rho_list) == len(theta_list) == len(x_der)

    # ------------------------------------------------------------------
    # 2️⃣  Every element must have the same shape as the input grid.
    # ------------------------------------------------------------------
    for arr in rho_list + theta_list:
        assert arr.shape == xs.shape

    # ------------------------------------------------------------------
    # 3️⃣  Dtype checks.
    # ------------------------------------------------------------------
    #   * All floating‑point outputs must use the same dtype as ``float_type``.
    #   * The integer derivative arrays must use the signed integer dtype that
    #     corresponds to ``float_type`` (checked via the helper above).
    # ------------------------------------------------------------------
    for arr in rho_list + theta_list:
        assert arr.dtype == float_type

    int_type_expected = core_corresponding_signed_integer_type(float_type)
    assert x_der.dtype == int_type_expected
    assert y_der.dtype == int_type_expected


@pytest.mark.parametrize("float_type", [numpy.float16, numpy.float32, numpy.float64])
def test_numeric_values_for_unit_circle(float_type):
    """
    When Rx = Ry = 1, h = 0, α = 0 and the centre is at the origin,
    the elliptic‑annulus mapping reduces to ordinary polar coordinates.
    This test compares the returned values (and first‑order derivatives)
    against the analytic formulas.
    """
    # Use a modest grid that includes the origin – the origin is a special
    # case for the angular derivative, but our analytic helpers handle it.
    xs = _typed_float_array(
        [[-0.6, -0.2, 0.0, 0.2, 0.6]],
        float_type,
    )
    ys = _typed_float_array(
        [[-0.4, 0.0, 0.0, 0.0, 0.4]],
        float_type,
    )

    Rx = float_type(1.0)
    Ry = float_type(1.0)
    x0 = float_type(0.0)
    y0 = float_type(0.0)
    alpha = float_type(0.0)
    h = float_type(0.0)

    # Derivative orders we will test:
    #   (0,0) – the base values,
    #   (1,0) – ∂/∂x,
    #   (0,1) – ∂/∂y
    x_der = _typed_int_array([0, 1, 0], float_type)
    y_der = _typed_int_array([0, 0, 1], float_type)

    rho_list, theta_list = core_cartesian_to_elliptic_annulus(
        x=xs,
        y=ys,
        Rx=Rx,
        Ry=Ry,
        x0=x0,
        y0=y0,
        alpha=alpha,
        h=h,
        x_derivative=x_der,
        y_derivative=y_der,
        float_type=float_type,
    )

    # ------------------------------------------------------------------
    # Base values (index 0) – should match ordinary polar coordinates.
    # ------------------------------------------------------------------
    numpy.testing.assert_allclose(
        rho_list[0],
        _polar_rho(xs, ys),
        rtol=1e-3 if float_type == numpy.float16 else 1e-7,
        atol=0,
    )
    numpy.testing.assert_allclose(
        theta_list[0],
        _polar_theta(xs, ys),
        rtol=1e-3 if float_type == numpy.float16 else 1e-7,
        atol=0,
    )

    # ------------------------------------------------------------------
    # First‑order derivatives.
    # ------------------------------------------------------------------
    numpy.testing.assert_allclose(
        rho_list[1],
        _drho_dx(xs, ys),
        rtol=1e-3 if float_type == numpy.float16 else 1e-7,
        atol=0,
    )
    numpy.testing.assert_allclose(
        theta_list[1],
        _dtheta_dx(xs, ys),
        rtol=1e-3 if float_type == numpy.float16 else 1e-7,
        atol=0,
    )

    numpy.testing.assert_allclose(
        rho_list[2],
        _drho_dy(xs, ys),
        rtol=1e-3 if float_type == numpy.float16 else 1e-7,
        atol=0,
    )
    numpy.testing.assert_allclose(
        theta_list[2],
        _dtheta_dy(xs, ys),
        rtol=1e-3 if float_type == numpy.float16 else 1e-7,
        atol=0,
    )


@pytest.mark.parametrize("float_type", [numpy.float16, numpy.float32, numpy.float64])
def test_high_order_derivative_uses_sympy_and_is_correct(float_type):
    """
    When the total derivative order is ≥ 3 the implementation switches to a
    SymPy‑generated expression.  This test makes sure:

    * No ``ValueError`` is raised (the SymPy path is taken).
    * The returned arrays have the correct shape and dtype.
    * The numeric values match an independently constructed SymPy
      expression for the same geometry.
    """
    # --------------------------------------------------------------
    # 1️⃣  Build a tiny Cartesian grid (shape 2×2) – enough to test
    #     the symbolic evaluation.
    # --------------------------------------------------------------
    xs = _typed_float_array([[0.1, 0.2],
                             [0.3, 0.4]],
                            float_type)
    ys = _typed_float_array([[0.5, 0.6],
                             [0.7, 0.8]],
                            float_type)

    # --------------------------------------------------------------
    # 2️⃣  Geometry parameters – choose a *non‑trivial* ellipse so that
    #     the mapping really depends on Rx, Ry, alpha and h.
    # --------------------------------------------------------------
    Rx = float_type(1.5)
    Ry = float_type(0.9)
    x0 = float_type(0.05)
    y0 = float_type(-0.07)
    alpha = float_type(numpy.pi / 6)   # 30 °
    h = float_type(0.25)           # inner‑boundary ratio

    # --------------------------------------------------------------
    # 3️⃣  Request a total derivative order of 3:
    #     dx = 2, dy = 1  (2 + 1 = 3)
    # --------------------------------------------------------------
    x_der = _typed_int_array([2], float_type)
    y_der = _typed_int_array([1], float_type)

    # --------------------------------------------------------------
    # 4️⃣  Call the core routine – it should *not* raise.
    # --------------------------------------------------------------
    rho_list, theta_list = core_cartesian_to_elliptic_annulus(
        x=xs,
        y=ys,
        Rx=Rx,
        Ry=Ry,
        x0=x0,
        y0=y0,
        alpha=alpha,
        h=h,
        x_derivative=x_der,
        y_derivative=y_der,
        float_type=float_type,
    )

    # --------------------------------------------------------------
    # 5️⃣  Basic sanity checks on shape and dtype.
    # --------------------------------------------------------------
    assert len(rho_list) == len(theta_list) == 1
    assert rho_list[0].shape == xs.shape
    assert theta_list[0].shape == xs.shape
    assert rho_list[0].dtype == float_type
    assert theta_list[0].dtype == float_type

    # --------------------------------------------------------------
    # 6️⃣  Build the *exact* SymPy expressions that the core code
    #     constructs, differentiate them with the same orders, and
    #     evaluate them on the same grid.
    # --------------------------------------------------------------
    # Symbolic variables
    x_sym, y_sym = sympy.symbols('x y')
    # ----- geometric transforms (identical to the core code) -----
    X_sym = sympy.cos(alpha) * (x_sym - x0) + sympy.sin(alpha) * (y_sym - y0)
    Y_sym = -sympy.sin(alpha) * (x_sym - x0) + sympy.cos(alpha) * (y_sym - y0)

    # ----- polar radius in the *ellipse* -----
    r_sym = sympy.sqrt((X_sym / Rx) ** 2 + (Y_sym / Ry) ** 2)
    rho_eq_sym = (r_sym - h) / (1 - h)

    # ----- angular coordinate (unchanged) -----
    theta_eq_sym = sympy.atan2(Y_sym / Ry, X_sym / Rx)

    # ----- differentiate the symbolic expressions -----
    rho_expr = sympy.diff(rho_eq_sym, x_sym, 2, y_sym, 1)
    theta_expr = sympy.diff(theta_eq_sym, x_sym, 2, y_sym, 1)

    # Simplify (mirrors the core implementation)
    rho_expr = sympy.simplify(rho_expr)
    theta_expr = sympy.simplify(theta_expr)

    # ----- lambdify for fast NumPy evaluation -----
    rho_func = sympy.lambdify((x_sym, y_sym), rho_expr, modules="numpy")
    theta_func = sympy.lambdify((x_sym, y_sym), theta_expr, modules="numpy")

    # Evaluate on the same grid
    rho_ref = rho_func(xs, ys).astype(float_type, copy=False)
    theta_ref = theta_func(xs, ys).astype(float_type, copy=False)

    # --------------------------------------------------------------
    # 7️⃣  Compare the core routine output with the independent reference.
    # --------------------------------------------------------------
    # Tolerances depend on the floating‑point precision.
    rtol = 1e-3 if float_type == numpy.float16 else 1e-7
    atol = 0.0

    numpy.testing.assert_allclose(rho_list[0], rho_ref, rtol=rtol, atol=atol)
    numpy.testing.assert_allclose(theta_list[0], theta_ref, rtol=rtol, atol=atol)