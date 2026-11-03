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

import math
from numbers import Integral, Real
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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
    r"""
    Check whether a value is an integer (Python or numpy), booleans excluded.

    Parameters
    ----------
    value : Any
        The value to check.

    Returns
    -------
    bool
        True if ``value`` is an integer and not a boolean.

    """
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
    return _broadcast_parsed({name: _as_integers(value, name) for name, value in values.items()})


def _broadcast_parsed(parsed: Dict[str, Any]) -> Tuple[Dict[str, List[Any]], bool]:
    r"""
    Broadcast already parsed parameters to lists of the same length.

    Each parameter is either a single value or a list (see :func:`_as_integers` and :func:`_as_choices`).
    The single values are repeated to the common length :math:`N` of the lists, and all the lists must
    have the same length.

    Parameters
    ----------
    parsed : Dict[str, Any]
        The parsed parameters (single values or lists), by name.

    Returns
    -------
    Dict[str, List[Any]]
        The parameters as lists of the same length, by name. If all the parameters are single values,
        the lists have length 1.

    bool
        True if all the parameters are single values.

    Raises
    ------
    ValueError
        If the lists do not have the same length.

    """
    lengths = {name: len(value) for name, value in parsed.items() if isinstance(value, list)}

    if not lengths:
        return {name: [value] for name, value in parsed.items()}, True

    if len(set(lengths.values())) > 1:
        details = ", ".join(f"{name}={length}" for name, length in lengths.items())
        raise ValueError(f"The sequences must have the same length (got {details}).")

    size = next(iter(lengths.values()))
    return {name: value if isinstance(value, list) else [value] * size for name, value in parsed.items()}, False


def _as_choices(value: Any, name: str, choices: Sequence[str]) -> Union[str, List[str]]:
    r"""
    Check a string parameter, or a sequence of string parameters, against the accepted choices.

    Parameters
    ----------
    value : Any
        A string, a sequence of strings (list or tuple) or a 1D numpy array of strings.

    name : str
        The name of the parameter, used in the error messages.

    choices : Sequence[str]
        The accepted strings.

    Returns
    -------
    Union[str, List[str]]
        A ``str`` if ``value`` is a single string, a ``List[str]`` otherwise.

    Raises
    ------
    TypeError
        If ``value`` is not a string, a sequence of strings or a 1D numpy array of strings.

    ValueError
        If a string is not in ``choices``.

    """
    if isinstance(value, str):
        items, single = [value], True
    elif isinstance(value, numpy.ndarray) and value.ndim == 1 and numpy.issubdtype(value.dtype, numpy.str_):
        items, single = [str(v) for v in value], False
    elif isinstance(value, Sequence) and not isinstance(value, bytes) and all(isinstance(v, str) for v in value):
        items, single = list(value), False
    else:
        raise TypeError(f"{name} must be a string, a sequence of strings or a 1D numpy array of strings.")

    for item in items:
        if item not in choices:
            raise ValueError(f"{name} must be one of {', '.join(repr(c) for c in choices)} (got {item!r}).")
    return items[0] if single else items


# Accepted names of the indexing conventions of the Zernike polynomials (case insensitive) -> canonical name
INDEXINGS = {"osa": "OSA", "ansi": "OSA", "osa/ansi": "OSA", "noll": "Noll", "fringe": "Fringe"}


def _as_indexing(indexing: Any) -> str:
    r"""
    Check and convert the name of an indexing convention of the Zernike polynomials to its canonical name.

    Parameters
    ----------
    indexing : Any
        The name of the indexing convention: ``"OSA"`` (or ``"ANSI"``), ``"Noll"`` or ``"Fringe"``, case insensitive.

    Returns
    -------
    str
        The canonical name ``"OSA"``, ``"Noll"`` or ``"Fringe"``.

    Raises
    ------
    TypeError
        If ``indexing`` is not a string.

    ValueError
        If ``indexing`` is not a known indexing convention.

    """
    if not isinstance(indexing, str):
        raise TypeError("indexing must be a string.")
    key = indexing.strip().lower()
    if key not in INDEXINGS:
        raise ValueError(f"indexing must be 'OSA' (or 'ANSI'), 'Noll' or 'Fringe' (got {indexing!r}).")
    return INDEXINGS[key]


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
    Check the consistency of the parameters of an elliptic annulus and convert them to Python floats.

    None is accepted for any parameter and replaced by its default value
    (:math:`R_x = R_y = 1`, :math:`x_0 = y_0 = \alpha = h = 0`, i.e. the unit disk).

    The elliptic annulus is well defined only if all the parameters are finite, the semi-axes are
    strictly positive and the ratio :math:`h` of the inner to the outer semi-axes is in :math:`[0, 1)`.

    Parameters
    ----------
    Rx : Any
        The semi-axis of the ellipse along its x axis. Must be a finite and strictly positive real number.

    Ry : Any
        The semi-axis of the ellipse along its y axis. Must be a finite and strictly positive real number.

    x0 : Any
        The x coordinate of the center of the ellipse. Must be a finite real number.

    y0 : Any
        The y coordinate of the center of the ellipse. Must be a finite real number.

    alpha : Any
        The rotation angle of the ellipse in radians. Must be a finite real number.

    h : Any
        The ratio of the inner to the outer semi-axes. Must be a real number in :math:`[0, 1)`.

    Returns
    -------
    Tuple[float, float, float, float, float, float]
        The parameters ``(Rx, Ry, x0, y0, alpha, h)``.

    Raises
    ------
    TypeError
        If a parameter is not a real number (booleans are not accepted).

    ValueError
        If a parameter is not finite, if ``Rx`` or ``Ry`` is not strictly positive, or if ``h`` is not
        in :math:`[0, 1)`.

    """
    defaults = {"Rx": 1.0, "Ry": 1.0, "x0": 0.0, "y0": 0.0, "alpha": 0.0, "h": 0.0}
    values = {"Rx": Rx, "Ry": Ry, "x0": x0, "y0": y0, "alpha": alpha, "h": h}

    parameters = {}
    for name, value in values.items():
        if value is None:
            value = defaults[name]
        if not isinstance(value, Real) or isinstance(value, bool):
            raise TypeError(f"{name} must be a real number.")
        try:
            value = float(value)
        except OverflowError as error:
            raise ValueError(f"{name} must be finite.") from error
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")
        parameters[name] = value

    if parameters["Rx"] <= 0 or parameters["Ry"] <= 0:
        raise ValueError("Rx and Ry must be strictly positive.")
    if not 0 <= parameters["h"] < 1:
        raise ValueError("h must be in the range [0, 1).")

    return tuple(parameters[name] for name in ("Rx", "Ry", "x0", "y0", "alpha", "h"))


def _as_order(value: Any, name: str = "order") -> int:
    r"""
    Check and convert a maximum radial order to a Python integer.

    Parameters
    ----------
    value : Any
        The value to convert. Must be a non-negative integer (Python or numpy, booleans excluded).

    name : str, optional
        The name of the parameter, used in the error messages. Default is ``"order"``.

    Returns
    -------
    int
        The order.

    Raises
    ------
    TypeError
        If ``value`` is not an integer.

    ValueError
        If ``value`` is negative.

    """
    if not _is_integer(value):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


def _as_default_value(value: Any, name: str = "default") -> float:
    r"""
    Check and convert the value used outside the domain of definition to a Python float.

    The accepted values are the integers and the floating point numbers (Python or numpy), including
    ``numpy.nan`` and ``numpy.inf``. Booleans are not accepted.

    Parameters
    ----------
    value : Any
        The value to convert.

    name : str, optional
        The name of the parameter, used in the error messages. Default is ``"default"``.

    Returns
    -------
    float
        The value as a Python float.

    Raises
    ------
    TypeError
        If ``value`` is not an integer or a floating point number.

    ValueError
        If ``value`` is an integer too large to be converted to a float.

    """
    if isinstance(value, (bool, numpy.bool_)) or not isinstance(value, (int, float, numpy.integer, numpy.floating)):
        raise TypeError(f"{name} must be an integer or a floating point number (numpy.nan and numpy.inf are accepted).")
    try:
        return float(value)
    except OverflowError as error:
        raise ValueError(f"{name} is too large to be converted to a floating point number.") from error


def _check_boolean(value: Any, name: str) -> bool:
    r"""
    Check that a parameter is a boolean.

    Parameters
    ----------
    value : Any
        The value to check.

    name : str
        The name of the parameter, used in the error message.

    Returns
    -------
    bool
        The value.

    Raises
    ------
    TypeError
        If ``value`` is not a boolean.

    """
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean.")
    return value


def _broadcast_indices(integers: Dict[str, Any], orders: Dict[str, Any]) -> Tuple[Dict[str, List[int]], bool]:
    r"""
    Convert the integer parameters and the derivative orders to lists of the same length.

    The derivative orders can be None (meaning 0) and must be non-negative. All the parameters are
    then broadcast with :func:`_broadcast_integers`.

    Parameters
    ----------
    integers : Dict[str, Any]
        The integer parameters (e.g. ``n`` and ``m``), by name.

    orders : Dict[str, Any]
        The derivative orders (e.g. ``rho_derivative``), by name. None is replaced by 0.

    Returns
    -------
    Dict[str, List[int]]
        All the parameters as lists of the same length, by name (``integers`` first, then ``orders``).

    bool
        True if all the parameters are single integers.

    Raises
    ------
    TypeError
        If a parameter is not an integer or a sequence of integers.

    ValueError
        If the sequences do not have the same length, or if a derivative order is negative.

    """
    orders = {name: 0 if value is None else value for name, value in orders.items()}
    indices, single = _broadcast_integers(**integers, **orders)
    for name in orders:
        if any(order < 0 for order in indices[name]):
            raise ValueError(f"{name} must contain non-negative integers.")
    return indices, single


def _format_output(output: numpy.ndarray, single: bool, to_list: bool) -> Union[numpy.ndarray, List[numpy.ndarray]]:
    r"""
    Format the array of shape ``(N, ...)`` returned by the public functions.

    Parameters
    ----------
    output : numpy.ndarray
        The computed values, with shape ``(N, ...)``.

    single : bool
        If True, only ``output[0]`` is returned (all the indices were single integers).

    to_list : bool
        If True (and ``single`` is False), a list of :math:`N` arrays is returned.

    Returns
    -------
    Union[numpy.ndarray, List[numpy.ndarray]]
        ``output[0]`` if ``single``, ``list(output)`` if ``to_list``, ``output`` otherwise.

    """
    if single:
        return output[0]
    if to_list:
        return list(output)
    return output


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
    resolution = _check_resolution(resolution)
    if not isinstance(rho_power, Real) or isinstance(rho_power, bool):
        raise TypeError("rho_power must be a real number.")
    if not numpy.isfinite(rho_power) or rho_power <= 0:
        raise ValueError("rho_power must be a strictly positive real number.")
    return resolution, float(rho_power)


def _check_resolution(resolution: Any) -> int:
    r"""
    Check the number of points along each axis of a display grid.

    Parameters
    ----------
    resolution : Any
        The number of points along each axis. Must be an integer greater than or equal to 2.

    Returns
    -------
    int
        The resolution.

    Raises
    ------
    TypeError
        If ``resolution`` is not an integer.

    ValueError
        If ``resolution`` is lower than 2.

    """
    if not _is_integer(resolution):
        raise TypeError("resolution must be an integer.")
    if resolution < 2:
        raise ValueError("resolution must be greater than or equal to 2.")
    return int(resolution)

def _check_total_derivative_order(
    first: List[int],
    second: List[int],
    maximum: int,
    first_name: str,
    second_name: str,
) -> None:
    r"""
    Check that the total order of each pair of derivatives does not exceed a maximum.

    Parameters
    ----------
    first : List[int]
        The orders of the derivatives with respect to the first variable.

    second : List[int]
        The orders of the derivatives with respect to the second variable, with the same length as ``first``.

    maximum : int
        The maximum total order allowed for each pair ``(first[i], second[i])``.

    first_name : str
        The name of the first parameter, used in the error message.

    second_name : str
        The name of the second parameter, used in the error message.

    Raises
    ------
    ValueError
        If ``first[i] + second[i] > maximum`` for an index ``i``.

    """
    for first_order, second_order in zip(first, second):
        if first_order + second_order > maximum:
            raise ValueError(
                f"The total order of derivation {first_name} + {second_name} must be lower than or equal to "
                f"{maximum} (got {first_name}={first_order} and {second_name}={second_order})."
            )


def _as_fit_weights(weights: Any, shape: Tuple[int, ...]) -> numpy.ndarray:
    r"""
    Convert the weights of a least squares fit to a floating point array.

    Parameters
    ----------
    weights : Any
        The weights, with the shape ``shape``. None means that all the weights are equal to 1.

    shape : Tuple[int, ...]
        The expected shape of the weights (the shape of the samples).

    Returns
    -------
    numpy.ndarray
        The weights as a floating point array of shape ``shape`` (``numpy.float64`` if ``weights`` is None).

    Raises
    ------
    TypeError
        If ``weights`` can not be converted to a floating point array.

    ValueError
        If ``weights`` does not have the shape ``shape``, or if a weight is negative.

    """
    if weights is None:
        return numpy.ones(shape, dtype=numpy.float64)
    weights = _as_floating_array(weights, "weights")
    if weights.shape != tuple(shape):
        raise ValueError(f"weights must have the same shape as the samples (got {weights.shape} instead of {tuple(shape)}).")
    if numpy.any(weights < 0):
        raise ValueError("weights must be non negative.")
    return weights


def _as_coefficients(coefficients: Any, name: str = "coefficients") -> numpy.ndarray:
    r"""
    Convert a vector of coefficients of Zernike polynomials to a 1D array of finite ``numpy.float64`` values.

    Parameters
    ----------
    coefficients : Any
        The coefficients. Must be convertible to a non-empty 1D array of finite real numbers.

    name : str, optional
        The name of the parameter, used in the error messages. Default is ``"coefficients"``.

    Returns
    -------
    numpy.ndarray
        A copy of the coefficients as a 1D array of ``numpy.float64``.

    Raises
    ------
    TypeError
        If ``coefficients`` can not be converted to an array of real numbers.

    ValueError
        If ``coefficients`` is not a non-empty 1D array, or contains non-finite values.

    """
    try:
        array = numpy.array(coefficients, dtype=numpy.float64)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be convertible to a 1D numpy array of real numbers.") from error
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D array (got shape {array.shape}).")
    if not numpy.all(numpy.isfinite(array)):
        raise ValueError(f"{name} must contain finite values.")
    return array


def _as_osa_coefficients(coefficients: Any, name: str = "coefficients") -> Tuple[numpy.ndarray, int]:
    r"""
    Convert a complete vector of coefficients in the OSA/ANSI ordering and find its maximum radial order.

    A complete vector contains the coefficients of all the polynomials with :math:`0 \leq n \leq N`, so its length
    must be :math:`P = \frac{(N+1)(N+2)}{2}` for an order :math:`N \geq 0` (1, 3, 6, 10, 15, ...).

    Parameters
    ----------
    coefficients : Any
        The coefficients in the OSA/ANSI ordering (see :func:`_as_coefficients`).

    name : str, optional
        The name of the parameter, used in the error messages. Default is ``"coefficients"``.

    Returns
    -------
    numpy.ndarray
        The coefficients as a 1D array of ``numpy.float64``.

    int
        The maximum radial order :math:`N`.

    Raises
    ------
    TypeError
        If ``coefficients`` can not be converted to an array of real numbers.

    ValueError
        If ``coefficients`` is not a non-empty 1D array of finite values, or if its length is not
        :math:`\frac{(N+1)(N+2)}{2}` for an integer :math:`N`.

    """
    array = _as_coefficients(coefficients, name)
    order = math.ceil((math.sqrt(8 * array.size + 1) - 3) / 2 - 1e-9)
    complete_size = (order + 1) * (order + 2) // 2
    if complete_size != array.size:
        raise ValueError(
            f"The length of {name} must be (N + 1)(N + 2) / 2 for a maximum order N in the OSA/ANSI ordering "
            f"(1, 3, 6, 10, 15, ...), got {array.size}. Pad the coefficients with zeros up to the order {order} "
            f"({complete_size} coefficients)."
        )
    return array, order


def _as_real(value: Any, name: str, minimum: Optional[float] = None, strict: bool = False) -> float:
    r"""
    Check that a parameter is a finite real number and convert it to a Python float.

    Parameters
    ----------
    value : Any
        The value to check. Booleans are not accepted.

    name : str
        The name of the parameter, used in the error messages.

    minimum : Optional[float], optional
        If given, the minimum value of the parameter. Default is None (no minimum).

    strict : bool, optional
        If True, the value must be strictly greater than ``minimum``. Default is False.

    Returns
    -------
    float
        The value.

    Raises
    ------
    TypeError
        If ``value`` is not a real number.

    ValueError
        If ``value`` is not finite, or is lower than (or equal to, if ``strict``) ``minimum``.

    """
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number.")
    try:
        value = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite.") from error
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    if minimum is not None and (value <= minimum if strict else value < minimum):
        raise ValueError(f"{name} must be {'strictly ' if strict else ''}greater than {'' if strict else 'or equal to '}{minimum}.")
    return value