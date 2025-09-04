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

from .__version__ import __version__
__all__ = ["__version__"]

from .core_polynomial import core_polynomial
from .core_symbolic import core_symbolic
from .core_display import core_display
__all__.extend(["core_polynomial", "core_symbolic", "core_display"])

from .radial_polynomial import radial_polynomial
R = radial_polynomial  # Alias
__all__.extend(["radial_polynomial", "R"])

from .zernike_polynomial import zernike_polynomial
Z = zernike_polynomial  # Alias
__all__.extend(["zernike_polynomial", "Z"])

from .radial_symbolic import radial_symbolic
from .zernike_symbolic import zernike_symbolic
__all__.extend(["radial_symbolic", "zernike_symbolic"])

from .radial_display import radial_display
from .zernike_display import zernike_display
__all__.extend(["radial_display", "zernike_display"])

from .xy_zernike_polynomial import xy_zernike_polynomial
Zxy = xy_zernike_polynomial  # Alias
__all__.extend(["xy_zernike_polynomial", "Zxy"])

from .zernike_order_to_index import zernike_order_to_index
from .zernike_index_to_order import zernike_index_to_order
__all__.extend(["zernike_order_to_index", "zernike_index_to_order"])

from .zernike_polynomial_up_to_order import zernike_polynomial_up_to_order
Zup = zernike_polynomial_up_to_order  # Alias
__all__.extend(["zernike_polynomial_up_to_order", "Zup"])

from .xy_zernike_polynomial_up_to_order import xy_zernike_polynomial_up_to_order
Zxyup = xy_zernike_polynomial_up_to_order  # Alias
__all__.extend(["xy_zernike_polynomial_up_to_order", "Zxyup"])