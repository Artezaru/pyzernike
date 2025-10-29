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

import numpy

def core_corresponding_signed_integer_type(
    float_type: type[numpy.floating]
) -> type[numpy.integer]:
    r"""
    Get the corresponding integer type for a given floating point type.

    .. warning::

        This method is a core function of ``pyzernike`` that is not designed to be use by the users directly.
        Please use the high level functions.

    Parameters
    ----------
    float_type : type[numpy.floating]
        The floating point type (e.g., numpy.float32, numpy.float64).

    Returns
    -------
    type[numpy.integer]
        The corresponding signed integer type.

    Raises
    ------
    ValueError
        If the provided type is not a supported floating point type.

    """
    # Fast assertion on the inputs
    assert issubclass(float_type, numpy.floating), "[pyzernike-core] float_type must be a numpy floating point type."

    # Map floating point types to corresponding signed integer types
    if float_type == numpy.float16:
        return numpy.int16
    if float_type == numpy.float32:
        return numpy.int32
    if float_type == numpy.float64:
        return numpy.int64
    else:
        raise ValueError(f"Unsupported floating point type: {float_type}, supported types are numpy.float16, numpy.float32 and numpy.float64 for pyzernike package.")