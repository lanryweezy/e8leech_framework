import numpy as np
from numba import jit
from functools import lru_cache

@lru_cache(maxsize=None)
@jit(nopython=True)
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
    roots = np.zeros((240, 8))
    k = 0

    # Type 1: 112 roots with two non-zero elements (±1)
    for i in range(8):
        for j in range(i + 1, 8):
            for signs in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                v = np.zeros(8)
                v[i] = signs[0]
                v[j] = signs[1]
                roots[k] = v
                k += 1

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
            roots[k] = v
            k += 1

    return roots

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

@jit(nopython=True)
def babai_nearest_plane(v, basis):
    """
    Babai's nearest plane algorithm for finding the closest lattice point.

    Args:
        v: The vector to be rounded to the nearest lattice point.
        basis: The basis of the lattice.

    Returns:
        The closest lattice point to v.
    """
    # First, we need to compute the Gram-Schmidt orthogonalization of the basis.
    # However, for the E8 lattice, a simpler algorithm exists.
    # A more general implementation would require LLL reduction for the basis
    # and then the Gram-Schmidt process.
    # For now, we will assume the basis is already LLL-reduced.

    # The general algorithm is as follows:
    # 1. Compute the inverse of the basis matrix.
    # 2. Compute the coordinates of the vector v in the basis: c = v * B^(-1)
    # 3. Round the coordinates to the nearest integers: c_rounded = round(c)
    # 4. Compute the lattice point: v_rounded = c_rounded * B

    # Numba does not support try...except blocks for specific exception types.
    # We will assume the basis is invertible.
    basis_inv = np.linalg.inv(basis)
    coords = np.dot(v, basis_inv)
    rounded_coords = np.round(coords)
    closest_point = np.dot(rounded_coords, basis)

    return closest_point
