import pytest
import numpy as np

from pyzernike.core import core_corresponding_signed_integer_type


@pytest.mark.parametrize(
    "float_type, expected_int_type",
    [
        (np.float16, np.int16),
        (np.float32, np.int32),
        (np.float64, np.int64),
    ],
)
def test_supported_float_types(float_type, expected_int_type):
    """Each supported floating‑point type should map to its signed integer counterpart."""
    result = core_corresponding_signed_integer_type(float_type)
    assert result is expected_int_type, (
        f"For {float_type!r} expected {expected_int_type!r} but got {result!r}"
    )


def test_unsupported_float_type():
    """Passing a non‑supported type should raise a ValueError."""
    # Choose a floating type that isn’t in the whitelist, e.g., float128 if available,
    # otherwise use a plain Python float (which is not a subclass of np.floating).
    unsupported_type = getattr(np, "float128", float)  # fallback to built‑in float

    with pytest.raises(ValueError) as exc_info:
        core_corresponding_signed_integer_type(unsupported_type)

    # Optional: verify the error message mentions the unsupported type.
    assert "Unsupported floating point type" in str(exc_info.value)