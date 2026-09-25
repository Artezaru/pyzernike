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

from numbers import Integral, Real
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy


def _as_floating_array(value: Any, name: str) -> numpy.ndarray:
    r"""
    Convert an array-like to a floating point numpy array.

    Floating point arrays are returned unchanged (their dtype is kept). Any other input is
    converted to ``numpy.float64``.

    Parameters
    ----------
    value : Any
        The array-like to convert.

    name : str
        The name of the parameter, used in the error message.

    Returns
    -------
    numpy.ndarray
        The floating point array.

    Raises
    ------
    TypeError
        If ``value`` can not be converted to a floating point array.

    """
    try:
        array = numpy.asarray(value)
        if not numpy.issubdtype(array.dtype, numpy.floating):
            array = array.astype(numpy.float64)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be convertible to a numpy array of floating point values.") from error
    return array


def _is_integer(value: Any) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _as_integers(value: Any, name: str) -> Union[int, List[int]]:
    r"""
    Convert an integer, a sequence of integers or a 1D numpy array of integers to Python integers.

    Parameters
    ----------
    value : Any
        The value to convert.

    name : str
        The name of the parameter, used in the error message.

    Returns
    -------
    Union[int, List[int]]
        An ``int`` if ``value`` is a single integer, a ``List[int]`` otherwise.

    Raises
    ------
    TypeError
        If ``value`` is not an integer, a sequence of integers or a 1D numpy array of integers.

    """
    if _is_integer(value):
        return int(value)

    if isinstance(value, numpy.ndarray):
        if value.ndim == 1 and numpy.issubdtype(value.dtype, numpy.integer):
            return value.tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if all(_is_integer(v) for v in value):
            return [int(v) for v in value]

    raise TypeError(f"{name} must be an integer, a sequence of integers or a 1D numpy array of integers.")


def _broadcast_integers(**values: Any) -> Tuple[Dict[str, List[int]], bool]:
    r"""
    Convert several integer parameters to lists of the same length.

    Each parameter can be a single integer or a sequence of integers (see :func:`_as_integers`).
    The single integers are repeated to the common length :math:`N` of the sequences, and all the
    sequences must have the same length.

    Parameters
    ----------
    **values : Any
        The parameters to convert, given by name.

    Returns
    -------
    Dict[str, List[int]]
        The parameters as lists of the same length, by name. If all the parameters are single
        integers, the lists have length 1.

    bool
        True if all the parameters are single integers.

    Raises
    ------
    TypeError
        If a parameter is not an integer or a sequence of integers.

    ValueError
        If the sequences do not have the same length.

    Examples
    --------
    .. code-block:: python

        _broadcast_integers(n=4, m=[0, 2, 4])
        # ({'n': [4, 4, 4], 'm': [0, 2, 4]}, False)

    """
    parsed = {name: _as_integers(value, name) for name, value in values.items()}
    lengths = {name: len(value) for name, value in parsed.items() if isinstance(value, list)}

    if not lengths:
        return {name: [value] for name, value in parsed.items()}, True

    if len(set(lengths.values())) > 1:
        details = ", ".join(f"{name}={length}" for name, length in lengths.items())
        raise ValueError(f"The sequences must have the same length (got {details}).")

    size = next(iter(lengths.values()))
    return {name: value if isinstance(value, list) else [value] * size for name, value in parsed.items()}, False


def _as_common_floating_arrays(**values: Any) -> List[numpy.ndarray]:
    r"""
    Convert several array-likes to floating point numpy arrays with the same shape and dtype.

    Each input is converted with :func:`_as_floating_array`. If the resulting dtypes differ, all the
    arrays are converted to ``numpy.float64``.

    Parameters
    ----------
    **values : Any
        The array-likes to convert, given by name.

    Returns
    -------
    List[numpy.ndarray]
        The floating point arrays, in the order of the parameters.

    Raises
    ------
    TypeError
        If a value can not be converted to a floating point array.

    ValueError
        If the arrays do not have the same shape.

    """
    arrays = [_as_floating_array(value, name) for name, value in values.items()]

    if len({array.shape for array in arrays}) > 1:
        details = ", ".join(f"{name}={array.shape}" for name, array in zip(values, arrays))
        raise ValueError(f"The arrays must have the same shape (got {details}).")

    if len({array.dtype for array in arrays}) > 1:
        arrays = [array.astype(numpy.float64) for array in arrays]

    return arrays


def _as_ellipse_parameters(
    Rx: Any,
    Ry: Any,
    x0: Any,
    y0: Any,
    alpha: Any,
    h: Any,
) -> Tuple[float, float, float, float, float, float]:
    r"""
    Check and convert the parameters of an elliptic annulus to Python floats.

    None is accepted for any parameter and replaced by its default value
    (:math:`R_x = R_y = 1`, :math:`x_0 = y_0 = \alpha = h = 0`, i.e. the unit disk).

    Parameters
    ----------
    Rx, Ry : Any
        The semi-axes of the ellipse. Must be strictly positive real numbers.

    x0, y0 : Any
        The center of the ellipse. Must be real numbers.

    alpha : Any
        The rotation angle of the ellipse in radians. Must be a real number.

    h : Any
        The ratio of the inner to the outer semi-axes. Must be a real number in :math:`[0, 1)`.

    Returns
    -------
    Tuple[float, float, float, float, float, float]
        The parameters ``(Rx, Ry, x0, y0, alpha, h)``.

    Raises
    ------
    TypeError
        If a parameter is not a real number.

    ValueError
        If ``Rx`` or ``Ry`` is not strictly positive, or if ``h`` is not in :math:`[0, 1)`.

    """
    defaults = {"Rx": 1.0, "Ry": 1.0, "x0": 0.0, "y0": 0.0, "alpha": 0.0, "h": 0.0}
    values = {"Rx": Rx, "Ry": Ry, "x0": x0, "y0": y0, "alpha": alpha, "h": h}

    parameters = {}
    for name, value in values.items():
        if value is None:
            value = defaults[name]
        if not isinstance(value, Real) or isinstance(value, bool):
            raise TypeError(f"{name} must be a real number.")
        parameters[name] = float(value)

    if parameters["Rx"] <= 0 or parameters["Ry"] <= 0:
        raise ValueError("Rx and Ry must be strictly positive.")
    if not 0 <= parameters["h"] < 1:
        raise ValueError("h must be in the range [0, 1).")

    return tuple(parameters[name] for name in ("Rx", "Ry", "x0", "y0", "alpha", "h"))


def _check_display_parameters(resolution: Any, rho_power: Any) -> Tuple[int, float]:
    r"""
    Check the parameters of the display functions.

    Parameters
    ----------
    resolution : Any
        The number of points along each axis. Must be an integer greater than or equal to 2.

    rho_power : Any
        The exponent :math:`p` of the display mapping :math:`\rho = s^{p}`. Must be a strictly positive real number.

    Returns
    -------
    Tuple[int, float]
        The parameters ``(resolution, rho_power)``.

    Raises
    ------
    TypeError
        If ``resolution`` is not an integer or ``rho_power`` is not a real number.

    ValueError
        If ``resolution`` is lower than 2 or ``rho_power`` is not strictly positive and finite.

    """
    if not _is_integer(resolution):
        raise TypeError("resolution must be an integer.")
    if not isinstance(rho_power, Real) or isinstance(rho_power, bool):
        raise TypeError("rho_power must be a real number.")
    if resolution < 2:
        raise ValueError("resolution must be greater than or equal to 2.")
    if not numpy.isfinite(rho_power) or rho_power <= 0:
        raise ValueError("rho_power must be a strictly positive real number.")
    return int(resolution), float(rho_power)