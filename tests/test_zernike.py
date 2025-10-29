import numpy as np
import pytest
import sympy

from pyzernike import radial_polynomial, zernike_polynomial


def test_polynomial_consistency():
    """Test that the zernike_polynomial function produces consistent results when called with multiple n, m, rho_derivative, and theta_derivative values."""
    list_n = []
    list_m = []
    list_rho_derivative = []
    list_theta_derivative = []

    for n in range(7):
        for m in range(0, n + 1):
            for rho_derivative in range(n):
                for theta_derivative in range(n):
                    list_n.append(n)
                    list_m.append(m)
                    list_rho_derivative.append(rho_derivative)
                    list_theta_derivative.append(theta_derivative)

    rho = np.linspace(0, 1, 100)
    theta = np.linspace(0, 2 * np.pi, 100)

    # Compute all in one
    common_result = zernike_polynomial(rho=rho, theta=theta, n=list_n, m=list_m, rho_derivative=list_rho_derivative, theta_derivative=list_theta_derivative)

    cumulative_result = []
    for i in range(len(list_n)):
        n = list_n[i]
        m = list_m[i]
        rho_derivative = list_rho_derivative[i]
        theta_derivative = list_theta_derivative[i]

        # Compute each one separately
        result = zernike_polynomial(rho=rho, theta=theta, n=[n], m=[m], rho_derivative=[rho_derivative], theta_derivative=[theta_derivative])[0]
        cumulative_result.append(result)

    # Check that all results are the same
    for i in range(len(cumulative_result)):
        assert np.allclose(cumulative_result[i], common_result[i], equal_nan=True), (
            f"Mismatch in cumulative results for index {i} with n={list_n[i]}, m={list_m[i]}, "
            f"rho_derivative={list_rho_derivative[i]}, theta_derivative={list_theta_derivative[i]}."
            f" Expected: {cumulative_result[i]}, Got: {common_result[i]}"
        )
    

def test_default_value():
    """ Test that the zernike_polynomial function correctly assigns the default value for out-of-domain rho values."""
    lx = np.linspace(-2, 2, 100)
    ly = np.linspace(-2, 2, 100)

    rho = np.sqrt(lx**2 + ly**2)
    theta = np.arctan2(ly, lx)
    Rho, Theta = np.meshgrid(rho, theta, indexing='ij')

    result = zernike_polynomial(Rho, Theta, n=[2], m=[0], rho_derivative=[0], theta_derivative=[0], default=-10.0)[0]
    assert np.all(result[Rho > 1] == -10.0), "Default value not correctly assigned for out-of-domain rho values."


def test_optional_derivatives():
    """ Test if rho and theta derivatives are set to 0 by default in zernike_polynomial function."""
    rho = np.linspace(0, 1, 100)
    theta = np.linspace(0, 2 * np.pi, 100)

    for rho_derivative in [None, [0,1,0]]:
        for theta_derivative in [None, [0,0,1]]:

            result = zernike_polynomial(rho=rho, theta=theta, n=[3, 2, 1], m=[1, 0, -1], rho_derivative=rho_derivative, theta_derivative=theta_derivative)

            result_explicit = zernike_polynomial(rho=rho, theta=theta, n=[3, 2, 1], m=[1, 0, -1], rho_derivative=rho_derivative if rho_derivative is not None else ([0]*len(theta_derivative) if theta_derivative is not None else [0, 0, 0]), theta_derivative=theta_derivative if theta_derivative is not None else ([0]*len(rho_derivative) if rho_derivative is not None else [0, 0, 0]))

            for i in range(len(result)):
                assert np.allclose(result[i], result_explicit[i], equal_nan=True), (
                    f"Mismatch for rho_derivative={rho_derivative}, theta_derivative={theta_derivative}. "
                    f"Expected: {result_explicit[i]}, Got: {result[i]}"
                )

def test_zernike_dimensions():
    """Test that the zernike_polynomial function returns results with the correct dimensions."""
    
    # Generate 100 random rho values between 0 and 1
    rho = np.linspace(0, 1, 10).astype(np.float64)
    theta = np.linspace(0, 2 * np.pi, 10).astype(np.float64)

    Rho, Theta = np.meshgrid(rho, theta, indexing='ij')

    # Test for different n and m values
    results = zernike_polynomial(Rho, Theta, n=[2, 5, 4], m=[0, 1, 2], rho_derivative=[0, 0, 0], theta_derivative=[0, 1, 0])

    assert len(results) == 3, "Expected 3 results for n=[2, 5, 4] and m=[0, 1, 2]."
    assert all(result.shape == Rho.shape for result in results), "Result shapes do not match input shape."
    assert not np.any(np.isnan(results)), "Results contain NaN values."

    # Consistency check with flattened inputs
    flat_rho = Rho.flatten()
    flat_theta = Theta.flatten()
    flat_results = zernike_polynomial(flat_rho, flat_theta, n=[2, 5, 4], m=[0, 1, 2], rho_derivative=[0, 0, 0], theta_derivative=[0, 1, 0])

    assert len(flat_results) == 3, "Expected 3 results for flattened inputs."
    assert all(result.shape == (Rho.size,) for result in flat_results), "Flattened result shapes do not match input shape."
    assert not np.any(np.isnan(flat_results)), "Flattened results contain NaN values."

    assert all(np.allclose(results[i].flatten(), flat_results[i]) for i in range(3)), "Mismatch in flattened results for n=[2, 5, 4] and m=[0, 1, 2]."


def test_radial_dimensions():
    """Test that the radial_polynomial function returns results with the correct dimensions."""
    # Generate 100 random rho values between 0 and 1
    Rho = np.linspace(0, 1, 100).reshape(20, 5)  # Reshape to a 2D array for consistency

    # Test for different n and m values
    results = radial_polynomial(Rho, n=[2, 3], m=[0, 1])
    assert len(results) == 2, "Expected 2 results for n=[2, 3] and m=[0, 1]."
    assert results[0].shape == Rho.shape, "Result shape does not match input shape for n=2, m=0."
    assert results[1].shape == Rho.shape, "Result shape does not match input shape for n=3, m=1."   

    # Consistency check with flattened inputs
    flat_Rho = Rho.flatten()
    flat_results = radial_polynomial(flat_Rho, n=[2, 3], m=[0, 1])
    assert len(flat_results) == 2, "Expected 2 results for flattened inputs."
    assert flat_results[0].shape == (Rho.size,), "Flattened result shape does not match input shape for n=2, m=0."
    assert flat_results[1].shape == (Rho.size,), "Flattened result shape does not match input shape for n=3, m=1."

    assert np.allclose(results[0].flatten(), flat_results[0]), "Mismatch in flattened results for n=2, m=0."
    assert np.allclose(results[1].flatten(), flat_results[1]), "Mismatch in flattened results for n=3, m=1."



def test_zernike_angular_consistency():
    """Test that the zernike_polynomial is 2pi periodic in theta."""
    rho = np.linspace(0, 1, 100)
    theta = np.linspace(0, 2 * np.pi, 100)
    theta_shifted = theta + 2 * np.pi

    for n in range(7):
        for m in range(-n, n + 1, 2):
            for dt in [0, 1, 2]:
                # Compute the Zernike polynomial for the original theta
                original_result = zernike_polynomial(rho=rho, theta=theta, n=[n], m=[m], rho_derivative=[0], theta_derivative=[dt])[0]

                # Compute the Zernike polynomial for the shifted theta
                shifted_result = zernike_polynomial(rho=rho, theta=theta_shifted, n=[n], m=[m], rho_derivative=[0], theta_derivative=[dt])[0]

                # Check if the results are equal
                assert np.allclose(original_result, shifted_result, equal_nan=True), (
                    f"Mismatch for n={n}, m={m}, theta_derivative={dt}. Expected: {original_result}, Got: {shifted_result}"
                )


def test_zernike_precompute_consistency():
    """Test that the zernike_polynomial function produces consistent results when precompute is True or False."""
    rho = np.linspace(0, 1, 100)
    theta = np.linspace(0, 2 * np.pi, 100)

    for n in range(7):
        for m in range(-n, n + 1, 2):
            for dr in range(3):
                for dt in range(3):
                    # Compute with precompute=True
                    result_precompute = zernike_polynomial(rho=rho, theta=theta, n=[n], m=[m], rho_derivative=[dr], theta_derivative=[dt], precompute=True)[0]

                    # Compute with precompute=False
                    result_no_precompute = zernike_polynomial(rho=rho, theta=theta, n=[n], m=[m], rho_derivative=[dr], theta_derivative=[dt], precompute=False)[0]

                    # Check if the results are equal
                    assert np.allclose(result_precompute, result_no_precompute, equal_nan=True), (
                        f"Mismatch for n={n}, m={m}, rho_derivative={dr}, theta_derivative={dt}. "
                        f"With precompute: {result_precompute}, Without precompute: {result_no_precompute}"
                    )


def test_zernike_dtype():
    """Test that the zernike_polynomial function returns results of the correct dtype."""
    rho_16 = np.linspace(0, 1, 100).astype(np.float16)
    theta_16 = np.linspace(0, 2 * np.pi, 100).astype(np.float16)
    rho_32 = np.linspace(0, 1, 100).astype(np.float32)
    theta_32 = np.linspace(0, 2 * np.pi, 100).astype(np.float32)
    rho_64 = np.linspace(0, 1, 100).astype(np.float64)
    theta_64 = np.linspace(0, 2 * np.pi, 100).astype(np.float64)

    for n in range(5):
        for m in range(-n, n + 1, 2):
            result_16 = zernike_polynomial(rho=rho_16, theta=theta_16, n=[n], m=[m], rho_derivative=[0], theta_derivative=[0])[0]
            result_32 = zernike_polynomial(rho=rho_32, theta=theta_32, n=[n], m=[m], rho_derivative=[0], theta_derivative=[0])[0]
            result_64 = zernike_polynomial(rho=rho_64, theta=theta_64, n=[n], m=[m], rho_derivative=[0], theta_derivative=[0])[0]

            assert result_16.dtype == np.float16, f"Expected dtype float16, got {result_16.dtype} for n={n}, m={m}"
            assert result_32.dtype == np.float32, f"Expected dtype float32, got {result_32.dtype} for n={n}, m={m}"
            assert result_64.dtype == np.float64, f"Expected dtype float64, got {result_64.dtype} for n={n}, m={m}"
