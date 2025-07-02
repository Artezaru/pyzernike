import numpy as np

for n in range(0, 50):
    for m in range(-n, n + 1, 2):
        j = int((n * (n + 2) + m) / 2)

        # Searching n and m from j
        n_search = int((-1 + np.sqrt(1 + 8 * j)) / 2)

        m_search = 2 * j - n_search * (n_search + 2)

        assert n_search == n, f"Search failed for n={n}, m={m}, j={j}. Found n_search={n_search}."
        assert m_search == m, f"Search failed for n={n}, m={m}, j={j}. Found m_search={m_search}."
        print(f"n={n}, m={m}, j={j} -> n_search={n_search}, m_search={m_search}")
