import pytest
import numpy

from pyzernike.core import core_create_precomputing_terms


# ----------------------------------------------------------------------
# Helper: map a float dtype to the matching signed integer dtype.
# ----------------------------------------------------------------------
_FLOAT_TO_INT = {
    numpy.float16: numpy.int16,
    numpy.float32: numpy.int32,
    numpy.float64: numpy.int64,
}


def _typed_int_array(values, float_type):
    """
    Return a 1‑D ``numpy.ndarray`` of the integer dtype that matches *float_type*.

    Parameters
    ----------
    values : iterable of ints
        The raw integer data.
    float_type : numpy.dtype
        One of ``np.float16``, ``np.float32`` or ``np.float64``.

    Returns
    -------
    numpy.ndarray
        1‑D array with dtype ``int16/int32/int64`` accordingly.
    """
    int_dtype = _FLOAT_TO_INT[float_type]
    return numpy.array(values, dtype=int_dtype)


# ----------------------------------------------------------------------
# Parameterised tests – run the same logic for float16, float32 and float64.
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "float_type",
    [numpy.float16, numpy.float32, numpy.float64],
)
def test_radial_mode_returns_correct_terms(float_type):
    """
    When ``flag_radial=True`` the function must:
      • accept integer inputs whose dtype matches the float_type,
      • return empty cosine/sine frequency arrays,
      • produce the correct power‑exponent and factorial sets,
      • report the proper max_n and max_m values.
    """
    # Simple valid data – all m = 0 so the angular part is irrelevant.
    n = _typed_int_array([0, 2, 4], float_type)
    m = _typed_int_array([0, 0, 0], float_type)
    dr = _typed_int_array([0, 0, 0], float_type)

    # No theta derivative needed for radial mode.
    result = core_create_precomputing_terms(
        n=n,
        m=m,
        rho_derivative=dr,
        theta_derivative=None,
        flag_radial=True,
        float_type=float_type,
    )

    powers, cos_freq, sin_freq, factorials, max_n, max_m = result

    # Cosine / sine frequencies must be empty.
    assert cos_freq.size == 0, f"Cosine frequencies should be empty in radial mode. Got: {cos_freq!r}"
    assert sin_freq.size == 0, f"Sine frequencies should be empty in radial mode. Got: {sin_freq!r}"

    # Expected powers: n - dr (k = 0 because m = 0 and dr = 0)
    expected_powers = {int(v) - int(d) for v, d in zip(n, dr)}
    assert set(powers.tolist()) == expected_powers

    # Factorial set must contain at least the numbers used in the algorithm.
    # We check that it is non‑empty and that the maximum n appears.
    assert factorials.size > 0
    assert int(max_n) in factorials

    # Max values must reflect the inputs.
    assert max_n == max(n)
    assert max_m == 0  # all m are zero

    # Assert that outputs have the correct dtypes.
    int_type = _FLOAT_TO_INT[float_type]
    assert powers.dtype == int_type, f"Powers array has incorrect dtype. Expected {int_type}, got {powers.dtype}."
    assert factorials.dtype == int_type, f"Factorials array has incorrect dtype. Expected {int_type}, got {factorials.dtype}."
    assert isinstance(max_n, int_type), f"max_n has incorrect type. Expected {int_type}, got {type(max_n)}."
    assert isinstance(max_m, int_type), f"max_m has incorrect type. Expected {int_type}, got {type(max_m)}."


@pytest.mark.parametrize(
    "float_type",
    [numpy.float16, numpy.float32, numpy.float64],
)
def test_full_mode_computes_cosine_and_sine_frequencies(float_type):
    """
    With ``flag_radial=False`` the function must also fill the cosine
    and sine frequency sets according to the parity rules described in the
    docstring.
    """
    n = _typed_int_array([2, 3, 4, 5], float_type)
    m = _typed_int_array([1, -2, 3, -4], float_type)
    dr = _typed_int_array([0, 0, 0, 0], float_type)
    dt = _typed_int_array([0, 1, 0, 1], float_type)   # theta derivatives

    powers, cos_freq, sin_freq, factorials, max_n, max_m = core_create_precomputing_terms(
        n=n,
        m=m,
        rho_derivative=dr,
        theta_derivative=dt,
        flag_radial=False,
        float_type=float_type,
    )

    # ------------------------------------------------------------------
    # Frequency checks – replicate the rule from the docstring:
    #   if (m>0 and dt even) or (m<0 and dt odd) -> cosine
    #   else if (m<0 and dt even) or (m>0 and dt odd) -> sine
    # ------------------------------------------------------------------
    expected_cos = {
        abs(int(m[i]))
        for i in range(len(m))
        if (int(m[i]) > 0 and int(dt[i]) % 2 == 0)
        or (int(m[i]) < 0 and int(dt[i]) % 2 == 1)
    }
    expected_sin = {
        abs(int(m[i]))
        for i in range(len(m))
        if (int(m[i]) < 0 and int(dt[i]) % 2 == 0)
        or (int(m[i]) > 0 and int(dt[i]) % 2 == 1)
    }

    assert set(cos_freq.tolist()) == expected_cos
    assert set(sin_freq.tolist()) == expected_sin

    # ------------------------------------------------------------------
    # Power / factorial sanity checks
    # ------------------------------------------------------------------
    # All returned powers must be within the range [-max_n, max_n].
    assert all(-max_n <= p <= max_n for p in powers)

    # Factorial set must contain the maximum n (used in the n‑k term).
    assert int(max_n) in factorials

    # Max values must match the inputs.
    assert max_n == max(n)
    assert max_m == max(abs(m))

    # Assert that outputs have the correct dtypes.
    int_type = _FLOAT_TO_INT[float_type]
    assert powers.dtype == int_type, f"Powers array has incorrect dtype. Expected {int_type}, got {powers.dtype}."
    assert factorials.dtype == int_type, f"Factorials array has incorrect dtype. Expected {int_type}, got {factorials.dtype}."
    assert cos_freq.dtype == int_type, f"Cosine frequencies array has incorrect dtype. Expected {int_type}, got {cos_freq.dtype}."
    assert sin_freq.dtype == int_type, f"Sine frequencies array has incorrect dtype. Expected {int_type}, got {sin_freq.dtype}."
    assert isinstance(max_n, int_type), f"max_n has incorrect type. Expected {int_type}, got {type(max_n)}."
    assert isinstance(max_m, int_type), f"max_m has incorrect type. Expected {int_type}, got {type(max_m)}."


def test_invalid_inputs_raise_assertions():
    """
    The internal fast‑assertions should fire for malformed arguments.
    """
    # Correctly typed arrays for a baseline call.
    n = numpy.array([2, 2], dtype=numpy.int32)
    m = numpy.array([1, 1], dtype=numpy.int32)
    dr = numpy.array([0, 0], dtype=numpy.int32)
    dt = numpy.array([0, 0], dtype=numpy.int32)

    # 1️⃣  Wrong float_type (not a NumPy floating type)
    with pytest.raises(AssertionError):
        core_create_precomputing_terms(
            n=n,
            m=m,
            rho_derivative=dr,
            theta_derivative=dt,
            flag_radial=False,
            float_type=int,          # invalid
        )

    # 2️⃣  Mismatched sizes between the arrays
    with pytest.raises(AssertionError):
        core_create_precomputing_terms(
            n=n,
            m=m,
            rho_derivative=numpy.array([0], dtype=numpy.int32),  # size 1
            theta_derivative=dt,
            flag_radial=False,
            float_type=numpy.float32,
        )

    # 3️⃣  Providing theta_derivative while flag_radial=True
    with pytest.raises(AssertionError):
        core_create_precomputing_terms(
            n=n,
            m=m,
            rho_derivative=dr,
            theta_derivative=dt,
            flag_radial=True,
            float_type=numpy.float32,
        )

    # 4️⃣  Supplying a non‑integer dtype for n (should fail the fast‑assertion)
    with pytest.raises(AssertionError):
        core_create_precomputing_terms(
            n=numpy.array([2.0, 2.0], dtype=numpy.float32),  # wrong dtype
            m=m,
            rho_derivative=dr,
            theta_derivative=dt,
            flag_radial=False,
            float_type=numpy.float32,
        )