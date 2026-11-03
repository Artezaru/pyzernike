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


def _require_optional_dependency(module: str, feature: str, extra: str) -> None:
    r"""
    Check that an optional dependency of ``pyzernike`` is installed.

    Only the absence of ``module`` itself is reported as a missing dependency: any other error raised while
    importing it (e.g. a broken installation, or a missing dependency of ``module``) is propagated unchanged, so
    that its real cause is not hidden.

    Parameters
    ----------
    module : str
        The name of the module to import (e.g. ``"sympy"``).

    feature : str
        The description of the features that need the module, used in the error message
        (e.g. ``"The symbolic functions"``).

    extra : str
        The name of the extra of ``pyzernike-polynomials`` that installs the module (e.g. ``"symbolic"``).

    Raises
    ------
    ImportError
        If ``module`` is not installed.

    """
    try:
        import_module(module)
    except ModuleNotFoundError as error:
        if error.name != module:
            raise
        raise ImportError(
            f"{feature} of pyzernike require the package '{module}', which is not installed in this Python "
            f"environment. Install it with:\n\n"
            f"    pip install {module}\n\n"
            f"(or install pyzernike with its optional dependencies: pip install \"pyzernike-polynomials[{extra}]\"). "
            f"If you have just installed it, restart the Python session."
        ) from error