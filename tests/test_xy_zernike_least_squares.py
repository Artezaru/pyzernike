import warnings

import numpy
import pytest

from pyzernike import (
    xy_zernike_evaluate,
    xy_zernike_fit,
    xy_zernike_least_squares,
    zernike_order_to_index,
    zernike_polynomials_count,
)


ORDER = 4
P = zernike_polynomials_count(ORDER)
TRUE_PUPIL = dict(x0=0.08, y0=-0.05, Rx=1.04, Ry=0.97, alpha=0.2)
ALL_PARAMETERS = ("x0", "y0", "Rx", "Ry", "alpha", "h")


def _coefficients():
    c = numpy.zeros(P)
    c[zernike_order_to_index(n=[4, 3, 3, 2, 2], m=[0, 1, -1, 2, 0])] = [0.5, 0.2, -0.1, 0.15, 0.3]
    return c


def _samples(h=0.0, noise=0.0, n_points=81, seed=0):
    x, y = numpy.meshgrid(numpy.linspace(-1.2, 1.2, n_points), numpy.linspace(-1.2, 1.2, n_points))
    values = xy_zernike_evaluate(_coefficients(), x, y, h=h, **TRUE_PUPIL)  # NaN outside the pupil
    values = values + numpy.random.default_rng(seed).normal(scale=noise, size=x.shape)
    return x, y, values


def _finite_differences(residual, q, eps=1e-7):
    return numpy.stack([(residual(q + eps * e) - residual(q - eps * e)) / (2 * eps) for e in numpy.eye(q.size)], axis=1)


def _gauss_newton(residual, jacobian, q, iterations=15):
    for _ in range(iterations):
        q = q - numpy.linalg.lstsq(jacobian(q), residual(q), rcond=None)[0]
    return q


# ====================================================================== Jacobian

@pytest.mark.parametrize("fit_coefficients", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("use_recursive", [False, True])
@pytest.mark.parametrize("h", [0.0, 0.3])
def test_jacobian_finite_differences(fit_coefficients, normalize, use_recursive, h):
    """The Jacobian matches the centered finite differences of the residuals, for all the parameters."""
    x, y, values = _samples(h=h, n_points=31)
    rng = numpy.random.default_rng(1)
    c = _coefficients() + 0.05 * rng.normal(size=P)
    pupil_q = numpy.array([0.05, -0.02, 1.0, 0.95, 0.15, max(h, 0.1)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        residual, jacobian = xy_zernike_least_squares(
            x, y, values, ORDER, ALL_PARAMETERS, fit_coefficients=fit_coefficients,
            coefficients=None if fit_coefficients else c, normalize=normalize, use_recursive=use_recursive,
        )
    q = numpy.concatenate([c, pupil_q]) if fit_coefficients else pupil_q
    J = jacobian(q)
    assert J.shape == (residual(q).size, q.size)
    assert numpy.max(numpy.abs(J - _finite_differences(residual, q))) < 1e-6 * numpy.max(numpy.abs(J))


def test_coefficient_columns_are_the_polynomials():
    """With a fixed annulus, the columns of the coefficients are the polynomials, and one Gauss-Newton step from zero
    gives the linear least squares solution of xy_zernike_fit."""
    x, y, values = _samples(noise=0.01)
    residual, jacobian = xy_zernike_least_squares(x, y, values, ORDER, (), fit_coefficients=True, **TRUE_PUPIL)
    q = _gauss_newton(residual, jacobian, numpy.zeros(P), iterations=1)
    assert numpy.allclose(q, xy_zernike_fit(x, y, values, ORDER, **TRUE_PUPIL), atol=1e-10)


# ====================================================================== estimation

@pytest.mark.parametrize("h", [0.0, 0.3])
def test_gauss_newton_recovers_the_pupil(h):
    """With known coefficients, the parameters of the pupil are recovered from a wavefront without noise."""
    x, y, values = _samples(h=h)
    parameters = ("x0", "y0", "Rx", "Ry", "alpha")
    residual, jacobian = xy_zernike_least_squares(x, y, values, ORDER, parameters, coefficients=_coefficients(), h=h)
    q = _gauss_newton(residual, jacobian, numpy.array([0.0, 0.0, 1.0, 1.0, 0.0]))
    assert numpy.allclose(q, [TRUE_PUPIL[name] for name in parameters], atol=1e-8)
    assert numpy.max(numpy.abs(residual(q))) < 1e-10


def test_scipy_least_squares():
    """The functions can be given directly to scipy.optimize.least_squares, with bounds."""
    optimize = pytest.importorskip("scipy.optimize")
    h = 0.3
    x, y, values = _samples(h=h, noise=0.005)
    parameters = ("x0", "y0", "Rx", "Ry", "alpha", "h")
    residual, jacobian = xy_zernike_least_squares(x, y, values, ORDER, parameters, coefficients=_coefficients())
    bounds = ([-1, -1, 1e-3, 1e-3, -numpy.pi, 0.0], [1, 1, 3, 3, numpy.pi, 0.95])
    result = optimize.least_squares(residual, [0.0, 0.0, 1.0, 1.0, 0.0, 0.2], jac=jacobian, bounds=bounds)
    assert result.success
    assert numpy.allclose(result.x, [TRUE_PUPIL[name] for name in parameters[:5]] + [h], atol=5e-3)


# ====================================================================== options

def test_samples_and_weights():
    """The residuals are computed at the valid samples only, and scaled by the square roots of the weights."""
    x, y, values = _samples(noise=0.01)
    valid = numpy.isfinite(values)
    weights = numpy.random.default_rng(2).uniform(0.5, 2.0, size=x.shape)
    weights[0, :] = 0.0
    q = numpy.array([0.0, 0.0])
    plain, _ = xy_zernike_least_squares(x, y, values, ORDER, ("x0", "y0"), coefficients=_coefficients(), Rx=1.04, Ry=0.97)
    weighted, _ = xy_zernike_least_squares(x, y, values, ORDER, ("x0", "y0"), coefficients=_coefficients(), Rx=1.04, Ry=0.97, weights=weights)
    assert plain(q).shape == (int(valid.sum()),)
    keep = weights[valid] > 0
    assert numpy.allclose(weighted(q), numpy.sqrt(weights[valid][keep]) * plain(q)[keep])


def test_extrapolate_false():
    """Without extrapolation, the samples outside the current annulus have zero residuals and zero rows."""
    x, y, values = _samples()
    common = dict(coefficients=_coefficients(), Rx=1.04, Ry=0.97, alpha=0.2)
    residual, jacobian = xy_zernike_least_squares(x, y, values, ORDER, ("x0", "y0"), extrapolate=False, **common)
    q = numpy.array([0.3, 0.0])  # shifted pupil: some samples are outside
    r, J = residual(q), jacobian(q)
    outside = r == 0.0
    assert outside.any() and numpy.all(J[outside] == 0.0)
    extrapolated, _ = xy_zernike_least_squares(x, y, values, ORDER, ("x0", "y0"), **common)
    assert numpy.allclose(r[~outside], extrapolated(q)[~outside])


def test_cache_and_copies():
    """The results do not depend on the order of the calls, and the returned arrays can be modified."""
    x, y, values = _samples(noise=0.01)
    residual, jacobian = xy_zernike_least_squares(x, y, values, ORDER, ("x0", "Rx"), coefficients=_coefficients())
    q1, q2 = numpy.array([0.0, 1.0]), numpy.array([0.05, 1.02])
    J1, r1 = jacobian(q1), residual(q1)
    r2 = residual(q2)
    r1[:] = 0.0
    J1[:] = 0.0
    assert numpy.array_equal(residual(q2), r2)
    assert not numpy.all(residual(q1) == 0.0) and not numpy.all(jacobian(q1) == 0.0)


def test_identifiability_warning():
    x, y, values = _samples()
    with pytest.warns(RuntimeWarning, match="not identifiable"):
        xy_zernike_least_squares(x, y, values, ORDER, ("x0", "Rx"), fit_coefficients=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        xy_zernike_least_squares(x, y, values, ORDER, ("x0", "Rx"), fit_coefficients=True, h=0.3)
        xy_zernike_least_squares(x, y, values, ORDER, ("h",), fit_coefficients=True)
        xy_zernike_least_squares(x, y, values, ORDER, ("x0", "Rx"), coefficients=_coefficients())


# ====================================================================== errors

def test_invalid_inputs():
    x, y, values = _samples(n_points=21)
    c = _coefficients()
    with pytest.raises(ValueError):
        xy_zernike_least_squares(x, y, values, ORDER, ("x0", "z0"), coefficients=c)
    with pytest.raises(ValueError):
        xy_zernike_least_squares(x, y, values, ORDER, ("x0", "x0"), coefficients=c)
    with pytest.raises(TypeError):
        xy_zernike_least_squares(x, y, values, ORDER, 3, coefficients=c)
    with pytest.raises(ValueError):
        xy_zernike_least_squares(x, y, values, ORDER, (), coefficients=c)  # no unknown
    with pytest.raises(ValueError):
        xy_zernike_least_squares(x, y, values, ORDER, ("x0",))  # missing coefficients
    with pytest.raises(ValueError):
        xy_zernike_least_squares(x, y, values, ORDER, ("h",), fit_coefficients=True, coefficients=c)
    with pytest.raises(ValueError):
        xy_zernike_least_squares(x, y, values, ORDER, ("x0",), coefficients=numpy.ones(10))  # order 3 != 4
    with pytest.raises(ValueError):
        xy_zernike_least_squares(x, y, values[:5], ORDER, ("x0",), coefficients=c)
    with pytest.raises(ValueError):
        xy_zernike_least_squares(x, y, numpy.full(x.shape, numpy.nan), ORDER, ("x0",), coefficients=c)
    with pytest.raises(TypeError):
        xy_zernike_least_squares(x, y, values, ORDER, ("x0",), coefficients=c, extrapolate=1)


def test_invalid_unknowns():
    x, y, values = _samples(n_points=21)
    residual, jacobian = xy_zernike_least_squares(x, y, values, ORDER, ("Rx", "h"), coefficients=_coefficients())
    for q in ([1.0], [1.0, 0.1, 0.0], [-1.0, 0.1], [1.0, 1.0], [numpy.nan, 0.1]):
        with pytest.raises(ValueError):
            residual(numpy.array(q))
        with pytest.raises(ValueError):
            jacobian(numpy.array(q))