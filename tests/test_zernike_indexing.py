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

import pytest

from pyzernike import zernike_index_to_order, zernike_order_to_index


# Reference orders (n, m) of the first indices (cosine for m > 0, sine for m < 0)
NOLL = [(0, 0), (1, 1), (1, -1), (2, 0), (2, -2), (2, 2), (3, -1), (3, 1), (3, -3), (3, 3), (4, 0),
        (4, 2), (4, -2), (4, 4), (4, -4), (5, 1), (5, -1), (5, 3), (5, -3), (5, 5), (5, -5), (6, 0)]
FRINGE = [(0, 0), (1, 1), (1, -1), (2, 0), (2, 2), (2, -2), (3, 1), (3, -1), (4, 0), (3, 3), (3, -3),
          (4, 2), (4, -2), (5, 1), (5, -1), (6, 0)]


def test_reference_tables():
    """Test the first Noll and Fringe indices against the reference tables."""
    assert [zernike_index_to_order(j, indexing="Noll") for j in range(1, 23)] == NOLL
    assert [zernike_index_to_order(j, indexing="Fringe") for j in range(1, 17)] == FRINGE
    assert zernike_order_to_index([n for n, _ in NOLL], [m for _, m in NOLL], indexing="Noll") == list(range(1, 23))
    assert zernike_order_to_index([n for n, _ in FRINGE], [m for _, m in FRINGE], indexing="Fringe") == list(range(1, 17))


@pytest.mark.parametrize("indexing", ["OSA", "ANSI", "Noll", "Fringe"])
def test_round_trip(indexing):
    """Test that zernike_index_to_order and zernike_order_to_index are inverse of each other."""
    orders = [(n, m) for n in range(61) for m in range(-n, n + 1, 2)]
    indices = zernike_order_to_index([n for n, _ in orders], [m for _, m in orders], indexing=indexing)
    assert len(set(indices)) == len(indices), "The indices are not unique."
    n, m = zernike_index_to_order(indices, indexing=indexing)
    assert list(zip(n, m)) == orders, f"Round trip failed for the {indexing} indexing."

    # Exact for huge indices
    first = 0 if indexing in ("OSA", "ANSI") else 1
    j = 10**40 + first
    assert zernike_order_to_index(*zernike_index_to_order(j, indexing=indexing), indexing=indexing) == j


def test_noll_parity():
    """Test that the even Noll indices correspond to m > 0 (cosine) and the odd ones to m < 0 (sine)."""
    for j in range(2, 500):
        n, m = zernike_index_to_order(j, indexing="Noll")
        if m != 0:
            assert (m > 0) == (j % 2 == 0), f"Wrong parity for the Noll index {j}."


def test_indexing_names():
    """Test that the names of the indexing conventions are case insensitive and that OSA is the default."""
    assert zernike_index_to_order(7) == zernike_index_to_order(7, indexing="osa") == zernike_index_to_order(7, indexing="ANSI")
    assert zernike_order_to_index(3, 1, indexing="NOLL") == zernike_order_to_index(3, 1, indexing="noll") == 8


def test_invalid_inputs():
    """Test that invalid inputs raise the expected errors."""
    with pytest.raises(ValueError):
        zernike_index_to_order(0, indexing="Noll")  # Noll starts at 1
    with pytest.raises(ValueError):
        zernike_index_to_order([1, 0], indexing="Fringe")  # Fringe starts at 1
    with pytest.raises(ValueError):
        zernike_index_to_order(3, indexing="Zemax")  # unknown convention
    with pytest.raises(TypeError):
        zernike_index_to_order(3, indexing=1)  # not a string
    with pytest.raises(ValueError):
        zernike_order_to_index(2, 1, indexing="Noll")  # invalid orders