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
from typing import List


def _core_normalization(n: List[int], m: List[int]) -> List[float]:
    r"""
    Compute the normalization factors of Noll of the Zernike polynomials without any check.

    .. warning::

        This method is an internal function of ``pyzernike`` that is not designed to be used by the users directly.
        Please use :func:`pyzernike.zernike_normalization`.

    The normalized polynomials :math:`N_{n}^{m} Z_{n}^{m}` are orthonormal on the unit disk for the inner product
    averaged over the disk:

    .. math::

        \frac{1}{\pi} \int_{0}^{2\pi} \int_{0}^{1} N_{n}^{m} Z_{n}^{m}(\rho, \theta) \, N_{n'}^{m'} Z_{n'}^{m'}(\rho, \theta) \, \rho \, d\rho \, d\theta = \delta_{nn'} \delta_{mm'}

    with:

    .. math::

        N_{n}^{m} = \sqrt{(2 - \delta_{m0})(n + 1)} = \begin{cases}
            \sqrt{n + 1} & \text{if } m = 0 \\
            \sqrt{2(n + 1)} & \text{if } m \neq 0
        \end{cases}

    Parameters
    ----------
    n : List[int]
        The radial orders. Must be a list of non-negative Python integers.

    m : List[int]
        The azimuthal frequencies. Must be a list of Python integers with the same length as ``n``.

    Returns
    -------
    List[float]
        The normalization factors :math:`N_{n}^{m}`.

    """
    return [math.sqrt((1 if m_i == 0 else 2) * (n_i + 1)) for n_i, m_i in zip(n, m)]