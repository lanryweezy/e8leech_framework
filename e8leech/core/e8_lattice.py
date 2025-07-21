import numpy as np

def generate_e8_roots():
    """
    Generates the 240 root vectors of the E8 lattice.

    The E8 root system consists of 240 vectors in 8-dimensional Euclidean space.
    These vectors can be described as follows:
    - 112 roots of the form (±1, ±1, 0, 0, 0, 0, 0, 0) with all possible permutations
      of coordinates and signs.
    - 128 roots of the form (±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2)
      where the number of negative signs is even.
    """
    roots = []

    # Type 1: 112 roots with two non-zero elements (±1)
    for i in range(8):
        for j in range(i + 1, 8):
            for signs in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                v = np.zeros(8)
                v[i] = signs[0]
                v[j] = signs[1]
                roots.append(v)

    # Type 2: 128 roots with all elements being ±1/2
    for i in range(2**8):
        v = np.zeros(8)
        neg_count = 0
        for j in range(8):
            if (i >> j) & 1:
                v[j] = 0.5
            else:
                v[j] = -0.5
                neg_count += 1

        if neg_count % 2 == 0:
            roots.append(v)

    return np.array(roots)

def get_e8_basis():
    """
    Returns a basis for the E8 lattice.
    """
    # This is one possible basis for E8
    return np.array([
        [2, 0, 0, 0, 0, 0, 0, 0],
        [-1, 1, 0, 0, 0, 0, 0, 0],
        [0, -1, 1, 0, 0, 0, 0, 0],
        [0, 0, -1, 1, 0, 0, 0, 0],
        [0, 0, 0, -1, 1, 0, 0, 0],
        [0, 0, 0, 0, -1, 1, 0, 0],
        [0, 0, 0, 0, 0, -1, 1, 0],
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    ])
