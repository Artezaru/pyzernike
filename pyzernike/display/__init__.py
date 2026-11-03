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

from ..core.core_dependencies import _require_optional_dependency

_require_optional_dependency("matplotlib", feature="The display functions", extra="display")

from .radial_display import radial_display  # noqa: E402
from .zernike_display import zernike_display  # noqa: E402
from .display_interactive import display_interactive  # noqa: E402
from .xy_zernike_display import xy_zernike_display  # noqa: E402
from .xy_display_interactive import xy_display_interactive  # noqa: E402

__all__ = [
    "radial_display",
    "zernike_display",
    "display_interactive",
    "xy_zernike_display",
    "xy_display_interactive",
]