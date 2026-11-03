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

from importlib import import_module
from typing import TYPE_CHECKING, Any, List

from .__version__ import __version__

# Numerical API (numpy only)
from .polynomials import (
    radial_polynomial,
    zernike_polynomial,
    zernike_polynomial_up_to_order,
    zernike_evaluate,
    xy_zernike_polynomial,
    xy_zernike_polynomial_up_to_order,
    xy_zernike_parameter_derivative,
    xy_zernike_evaluate,
)
from .tools import (
    zernike_index_to_order,
    zernike_order_to_index,
    zernike_polynomials_count,
    zernike_normalization,
    cartesian_to_elliptic_annulus,
)
from .coefficients import (
    zernike_coefficients_convert,
    zernike_coefficients_rotate,
    zernike_coefficients_transform,
    zernike_rms,
    zernike_peak_to_valley,
    zernike_strehl_ratio,
)
from .fit import zernike_fit, xy_zernike_fit, xy_zernike_least_squares

# Optional API, imported on first access (matplotlib for display, sympy for symbolic)
_LAZY_SUBPACKAGES = ("display", "symbolic")

_LAZY_FUNCTIONS = {
    "radial_display": "display",
    "zernike_display": "display",
    "display_interactive": "display",
    "xy_zernike_display": "display",
    "xy_display_interactive": "display",
    "radial_symbolic": "symbolic",
    "zernike_symbolic": "symbolic",
}

if TYPE_CHECKING:  # only for the IDEs and the type checkers, never executed
    from .display import (
        radial_display,
        zernike_display,
        display_interactive,
        xy_zernike_display,
        xy_display_interactive,
    )
    from .symbolic import radial_symbolic, zernike_symbolic


def __getattr__(name: str) -> Any:
    r"""
    Import the optional subpackages and their functions on first access.

    Parameters
    ----------
    name : str
        The name of the attribute.

    Returns
    -------
    Any
        The subpackage or the function.

    Raises
    ------
    AttributeError
        If ``name`` is not an attribute of ``pyzernike``.

    ImportError
        If the optional dependency of the subpackage (matplotlib or sympy) is not installed.

    """
    if name in _LAZY_SUBPACKAGES:
        value = import_module(f".{name}", __name__)
    elif name in _LAZY_FUNCTIONS:
        value = getattr(import_module(f".{_LAZY_FUNCTIONS[name]}", __name__), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value  # cached: __getattr__ is not called again for this name
    return value


def __dir__() -> List[str]:
    r"""
    List the attributes of ``pyzernike``, including the optional ones not yet imported.

    Returns
    -------
    List[str]
        The sorted names.

    """
    return sorted(set(globals()) | set(_LAZY_SUBPACKAGES) | set(_LAZY_FUNCTIONS))


__all__ = [
    "__version__",
    # polynomials
    "radial_polynomial",
    "zernike_polynomial",
    "zernike_polynomial_up_to_order",
    "zernike_evaluate",
    "xy_zernike_polynomial",
    "xy_zernike_polynomial_up_to_order",
    "xy_zernike_parameter_derivative",
    "xy_zernike_evaluate",
    # tools
    "zernike_index_to_order",
    "zernike_order_to_index",
    "zernike_polynomials_count",
    "zernike_normalization",
    "cartesian_to_elliptic_annulus",
    # coefficients
    "zernike_coefficients_convert",
    "zernike_coefficients_rotate",
    "zernike_coefficients_transform",
    "zernike_rms",
    "zernike_peak_to_valley",
    "zernike_strehl_ratio",
    # fit
    "zernike_fit",
    "xy_zernike_fit",
    "xy_zernike_least_squares",
    # display (matplotlib)
    "radial_display",
    "zernike_display",
    "display_interactive",
    "xy_zernike_display",
    "xy_display_interactive",
    # symbolic (sympy)
    "radial_symbolic",
    "zernike_symbolic",
]