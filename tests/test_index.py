from pyzernike import zernike_order_to_index, zernike_index_to_order
import pytest



def test_zernike_order_to_index():
    """Test the conversion of Zernike orders to indices."""

    assert 0 == zernike_order_to_index([0], [0])[0], "Zernike order (0, 0) should map to index 0."
    assert 1 == zernike_order_to_index([1], [-1])[0], "Zernike order (1, -1) should map to index 1."
    assert 2 == zernike_order_to_index([1], [1])[0], "Zernike order (1, 1) should map to index 2."
    assert 3 == zernike_order_to_index([2], [-2])[0], "Zernike order (2, -2) should map to index 3."
    assert 4 == zernike_order_to_index([2], [0])[0], "Zernike order (2, 0) should map to index 4."
    assert 5 == zernike_order_to_index([2], [2])[0], "Zernike order (2, 2) should map to index 5."

def test_zernike_index_to_order():
    """Test the conversion of Zernike indices to orders."""
    
    n = [0, 1, 1, 2, 2, 2]
    m = [0, -1, 1, -2, 0, 2]
    indices = [0, 1, 2, 3, 4, 5]

    n_result, m_result = zernike_index_to_order(indices)

    assert n_result == n, f"Expected radial orders {n}, but got {n_result}."
    assert m_result == m, f"Expected azimuthal orders {m}, but got {m_result}."

def test_zernike_order_to_index_and_back():
    """Test the round-trip conversion of Zernike orders to indices and back."""
    
    for n in range(10):
        for m in range(-n, n + 1, 2):
            
            j = zernike_order_to_index([n], [m])[0]
            n_search, m_search = zernike_index_to_order([j])
            n_search = n_search[0]
            m_search = m_search[0]

            assert n_search == n, f"Search failed for n={n}, m={m}, j={j}. Found n_search={n_search}."
            assert m_search == m, f"Search failed for n={n}, m={m}, j={j}. Found m_search={m_search}."