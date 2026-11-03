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

import pytest
import numpy as np

from pyzernike import xy_zernike_parameter_derivative, xy_zernike_polynomial


PARAMETERS = ["x0", "y0", "Rx", "Ry", "alpha", "h"]


def _random_points(n_points=2000, seed=0):
    """Random points covering a rotated and shifted ellipse, with the exact center (x0, y0) as first point."""
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(-1.6, 1.6, (2, n_points))
    x[0], y[0] = 0.2, -0.1
    return x, y


@pytest.mark.parametrize("h", [0.0, 0.3])
@pytest.mark.parametrize("parameter", PARAMETERS)
def test_parameter_derivative_finite_differences(parameter, h):
    """Test that the derivatives with respect to the parameters of the annulus match centered finite differences of xy_zernike_polynomial."""
    x, y = _random_points()
    ellipse = dict(Rx=1.6, Ry=1.1, x0=0.2, y0=-0.1, alpha=0.5, h=h)
    epsilon = 1e-6

    for n in range(7):
        for m in range(-n, n + 1, 2):
            result = xy_zernike_parameter_derivative(x=x, y=y, n=n, m=m, parameter=parameter, **ellipse)

            ellipse_plus, ellipse_minus = dict(ellipse), dict(ellipse)
            if parameter == "h" and h == 0.0:
                # h must stay in [0, 1): right finite difference at h = 0
                ellipse_plus["h"], ellipse_minus["h"], step = epsilon, 0.0, epsilon
            else:
                ellipse_plus[parameter] += epsilon
                ellipse_minus[parameter] -= epsilon
                step = 2 * epsilon

            finite_difference = (
                xy_zernike_polynomial(x=x, y=y, n=n, m=m, **ellipse_plus)
                - xy_zernike_polynomial(x=x, y=y, n=n, m=m, **ellipse_minus)
            ) / step

            # Points crossing the boundary of the annulus between the two evaluations are not compared
            valid = np.isfinite(finite_difference) & np.isfinite(result)
            tolerance = 1e-4 if (parameter == "h" and h == 0.0) else 1e-6
            assert np.allclose(result[valid], finite_difference[valid], rtol=tolerance, atol=tolerance), (
                f"Mismatch with the finite differences for parameter={parameter}, h={h}, n={n}, m={m}. "
                f"Max error: {np.max(np.abs(result[valid] - finite_difference[valid]))}"
            )


def test_parameter_derivative_analytic():
    """Test the derivatives against analytic expressions for Z_1^1 = u and Z_2^0 = 2 (u^2 + v^2) - 1 (alpha = 0, h = 0)."""
    Rx, Ry, x0, y0 = 2.0, 1.5, 0.3, -0.4
    x = np.linspace(x0 - 1.0, x0 + 1.0, 50)
    y = np.linspace(y0 - 1.0, y0 + 1.0, 50)
    X, Y = np.meshgrid(x, y, indexing='ij')
    u, v = (X - x0) / Rx, (Y - y0) / Ry
    inside = u**2 + v**2 <= 1

    expected = {
        (1, 1): {"x0": -np.ones_like(u) / Rx, "y0": np.zeros_like(u), "Rx": -u / Rx, "Ry": np.zeros_like(u), "alpha": Ry / Rx * v},
        (2, 0): {"x0": -4 * u / Rx, "y0": -4 * v / Ry, "Rx": -4 * u**2 / Rx, "Ry": -4 * v**2 / Ry, "alpha": 4 * u * v * (Ry / Rx - Rx / Ry)},
    }

    for (n, m), derivatives in expected.items():
        for parameter, value in derivatives.items():
            result = xy_zernike_parameter_derivative(X, Y, n=n, m=m, parameter=parameter, Rx=Rx, Ry=Ry, x0=x0, y0=y0)
            assert np.allclose(result[inside], value[inside]), (
                f"Mismatch with the analytic derivative for n={n}, m={m}, parameter={parameter}."
            )


def test_parameter_derivative_translation():
    """Test that the derivatives with respect to the center are the opposite of the derivatives with respect to x and y."""
    x, y = _random_points()
    ellipse = dict(Rx=1.6, Ry=1.1, x0=0.2, y0=-0.1, alpha=0.5, h=0.0)

    for n in range(7):
        for m in range(-n, n + 1, 2):
            dx0, dy0 = xy_zernike_parameter_derivative(x=x, y=y, n=n, m=m, parameter=["x0", "y0"], **ellipse)
            dx, dy = xy_zernike_polynomial(x=x, y=y, n=[n, n], m=[m, m], x_derivative=[1, 0], y_derivative=[0, 1], **ellipse)

            assert np.allclose(dx0, -dx, equal_nan=True), f"d/dx0 is not -d/dx for n={n}, m={m}."
            assert np.allclose(dy0, -dy, equal_nan=True), f"d/dy0 is not -d/dy for n={n}, m={m}."


def test_parameter_derivative_center():
    """Test that the derivatives are finite at the center of a full ellipse (h = 0)."""
    x = np.array([0.2])
    y = np.array([-0.1])

    for n in range(8):
        for m in range(-n, n + 1, 2):
            result = xy_zernike_parameter_derivative(
                x=x, y=y, n=n, m=m, parameter=PARAMETERS, Rx=1.6, Ry=1.1, x0=0.2, y0=-0.1, alpha=0.5, h=0.0,
            )
            assert all(np.all(np.isfinite(value)) for value in result), (
                f"Non finite derivative at the center of the ellipse for n={n}, m={m}."
            )


def test_parameter_derivative_dimensions():
    """Test that the xy_zernike_parameter_derivative function returns results with the correct dimensions."""
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y, indexing='ij')

    results = xy_zernike_parameter_derivative(X, Y, n=[2, 5, 4], m=[0, 1, 2], parameter=["x0", "Rx", "h"], Rx=np.sqrt(2), Ry=np.sqrt(2))

    assert isinstance(results, list) and len(results) == 3, "Expected 3 results for n=[2, 5, 4] and m=[0, 1, 2]."
    assert all(result.shape == X.shape for result in results), "Result shapes do not match input shape."
    assert not np.any(np.isnan(results)), "Results contain NaN values."

    # Consistency check with flattened inputs
    flat_results = xy_zernike_parameter_derivative(X.flatten(), Y.flatten(), n=[2, 5, 4], m=[0, 1, 2], parameter=["x0", "Rx", "h"], Rx=np.sqrt(2), Ry=np.sqrt(2))
    assert all(result.shape == (X.size,) for result in flat_results), "Flattened result shapes do not match input shape."
    assert all(np.allclose(results[i].flatten(), flat_results[i]) for i in range(3)), "Mismatch in flattened results."

    # Single values and array output
    single = xy_zernike_parameter_derivative(X, Y, n=2, m=0, parameter="Rx")
    assert isinstance(single, np.ndarray) and single.shape == X.shape, "Expected a single array for single values."

    gradient = xy_zernike_parameter_derivative(X, Y, n=2, m=0, parameter=PARAMETERS, to_list=False)
    assert gradient.shape == (6,) + X.shape, "Expected an array of shape (6, ...) for the full gradient."

    parameters_array = xy_zernike_parameter_derivative(X, Y, n=2, m=0, parameter=np.array(["x0", "h"]), to_list=False)
    assert parameters_array.shape == (2,) + X.shape, "Expected an array of shape (2, ...) for a numpy array of parameters."


def test_parameter_derivative_precompute():
    """Test that the results do not depend on the precompute option."""
    x, y = _random_points()
    kwargs = dict(n=[4, 5, 6, 3], m=[2, -1, 0, 3], parameter=["alpha", "Ry", "h", "x0"], Rx=1.6, Ry=1.1, alpha=0.5, h=0.2, to_list=False)

    result_precompute = xy_zernike_parameter_derivative(x, y, precompute=True, **kwargs)
    result_no_precompute = xy_zernike_parameter_derivative(x, y, precompute=False, **kwargs)
    assert np.array_equal(result_precompute, result_no_precompute, equal_nan=True), "Results depend on the precompute option."


def test_invalid_polynomials():
    """Test that the derivatives of the invalid polynomials (|m| > n or n - m odd) are zero inside the annulus."""
    x, y = _random_points()

    for n, m in [(3, 0), (2, 4), (4, -1)]:
        for parameter in PARAMETERS:
            result = xy_zernike_parameter_derivative(x, y, n=n, m=m, parameter=parameter, Rx=1.6, Ry=1.1, h=0.2)
            assert np.all(result[np.isfinite(result)] == 0.0), f"Non zero derivative for the invalid polynomial n={n}, m={m}."


def test_default_value():
    """Test that the xy_zernike_parameter_derivative function correctly assigns the default value outside the annulus."""
    lx = np.linspace(-2, 2, 100)
    ly = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(lx, ly, indexing='ij')
    Rho = np.sqrt(X**2 + Y**2)

    for parameter in PARAMETERS:
        result = xy_zernike_parameter_derivative(X, Y, n=2, m=0, parameter=parameter, h=0.3, default=-10.0)
        assert np.all(result[Rho > 1] == -10.0), f"Default value not correctly assigned outside the annulus for {parameter}."
        assert np.all(result[Rho < 0.3] == -10.0), f"Default value not correctly assigned inside the inner ellipse for {parameter}."

        result_nan = xy_zernike_parameter_derivative(X, Y, n=2, m=0, parameter=parameter)
        assert np.all(np.isnan(result_nan[Rho > 1])), f"Default value is not NaN outside the annulus for {parameter}."


def test_parameter_derivative_dtype():
    """Test that the xy_zernike_parameter_derivative function returns results of the correct dtype."""
    x_32 = np.linspace(0, 1, 100).astype(np.float32)
    y_32 = np.linspace(0, 1, 100).astype(np.float32)
    x_64 = np.linspace(0, 1, 100).astype(np.float64)
    y_64 = np.linspace(0, 1, 100).astype(np.float64)

    for n in range(5):
        for m in range(-n, n + 1, 2):
            for parameter in PARAMETERS:
                result_32 = xy_zernike_parameter_derivative(x=x_32, y=y_32, n=n, m=m, parameter=parameter, Rx=2.0, Ry=2.0)
                result_64 = xy_zernike_parameter_derivative(x=x_64, y=y_64, n=n, m=m, parameter=parameter, Rx=2.0, Ry=2.0)

                assert result_32.dtype == np.float32, f"Expected dtype float32, got {result_32.dtype} for n={n}, m={m}, {parameter}"
                assert result_64.dtype == np.float64, f"Expected dtype float64, got {result_64.dtype} for n={n}, m={m}, {parameter}"


def test_invalid_inputs():
    """Test that invalid inputs raise the expected errors."""
    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 10)

    with pytest.raises(ValueError):
        xy_zernike_parameter_derivative(x, y, n=2, m=0, parameter="R")  # unknown parameter
    with pytest.raises(TypeError):
        xy_zernike_parameter_derivative(x, y, n=2, m=0, parameter=3)  # not a string
    with pytest.raises(TypeError):
        xy_zernike_parameter_derivative(x, y, n=2, m=0, parameter=["x0", 1])  # not a sequence of strings
    with pytest.raises(ValueError):
        xy_zernike_parameter_derivative(x, y, n=[2, 3], m=0, parameter=["x0", "y0", "h"])  # different lengths
    with pytest.raises(ValueError):
        xy_zernike_parameter_derivative(x, y, n=2, m=0, parameter="h", h=1.0)  # h out of [0, 1)
    with pytest.raises(ValueError):
        xy_zernike_parameter_derivative(x, y, n=2, m=0, parameter="Rx", Rx=np.inf)  # non finite parameter
    with pytest.raises(TypeError):
        xy_zernike_parameter_derivative(x, y, n=2, m=0, parameter="Rx", default=True)  # boolean default