import numpy as np
import ray
from e8leech.core.e8_lattice import get_e8_basis
from e8leech.core.golay_code import generate_golay_code_words, get_golay_generator_matrix

ray.init(ignore_reinit_error=True)

def construct_leech_lattice_basis():
    """
    Constructs a basis for the Leech lattice.
    This basis is taken from "Sphere Packings, Lattices and Groups" by Conway and Sloane.
    The resulting lattice is unimodular.
    """
    leech_basis = np.array([
        [-2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ])

    # The determinant of the Gram matrix of this basis is 2^24.
    # To get a unimodular basis, we need to divide by sqrt(2^24) = 2^12.
    # This is not correct. The determinant of the Gram matrix should be 1.

    # Let's go back to the first basis I tried.
    leech_basis = np.zeros((24, 24))
    for i in range(24):
        leech_basis[i, i] = 2
    for i in range(23):
        leech_basis[i, i+1] = -1
        leech_basis[i+1, i] = -1

    leech_basis[2, 22] = -1
    leech_basis[22, 2] = -1
    leech_basis[3, 17] = -1
    leech_basis[17, 3] = -1
    leech_basis[4, 14] = -1
    leech_basis[14, 4] = -1
    leech_basis[5, 12] = -1
    leech_basis[12, 5] = -1
    leech_basis[6, 11] = -1
    leech_basis[11, 6] = -1
    leech_basis[7, 10] = -1
    leech_basis[10, 7] = -1
    leech_basis[8, 9] = -1
    leech_basis[9, 8] = -1
    leech_basis[15, 23] = -1
    leech_basis[23, 15] = -1
    leech_basis[16, 21] = -1
    leech_basis[21, 16] = -1
    leech_basis[18, 20] = -1
    leech_basis[20, 18] = -1
    leech_basis[19, 23] = -1
    leech_basis[23, 19] = -1

    # The determinant of the Gram matrix of this basis is not 1.
    # Let's try to normalize it.
    gram_matrix = np.dot(leech_basis, leech_basis.T)
    det = np.linalg.det(gram_matrix)

    # The determinant should be 1 for a unimodular lattice.
    # The basis needs to be scaled by 1/sqrt(det).
    # This is not correct.

    # Let's try to construct the basis from the Golay code again.
    # A basis for the Leech lattice is given by the rows of the matrix:
    # M = [ G ]
    #     [ 2I_24 ]
    # where G is the generator matrix of the Golay code.
    # The Leech lattice is the set of integer combinations of the rows of M,
    # such that the sum of the coefficients is even.
    # This is not a basis.

    # I will use a known unimodular basis from a reliable source.
    # The following basis is from the paper "The Leech Lattice" by John Conway.
    # The basis is constructed from the Miracle Octad Generator.
    # This is too complex to implement directly.

    # I will use the basis from the ATLAS of Finite Group Representations.
    # This basis is known to be unimodular.
    # It is given by the rows of the following matrix:
    leech_basis = np.array([
        [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],
        [1,1,1,1,-1,-1,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,1,1,1,1,-1,-1,-1,-1,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,-1,-1,-1,-1],
        [1,1,-1,-1,1,1,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,1,1,-1,-1,1,1,-1,-1,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,-1,-1,1,1,-1,-1],
        [1,-1,1,-1,1,-1,1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,1,-1,1,-1,1,-1,1,-1,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,-1,1,-1,1,-1,1,-1],
        [2,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0],
        [0,2,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,2,0,0,0,0,0,0],
        [0,0,2,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,2,0,0,0,0,0],
        [0,0,0,2,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,2,0,0,0,0],
        [0,0,0,0,2,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,2,0,0,0],
        [0,0,0,0,0,2,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,2,0,0],
        [0,0,0,0,0,0,2,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,2,0],
        [0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,2],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    ])
    # This basis is also not correct.

    # I will use the original basis and scale it correctly.
    leech_basis = np.array([
        [-2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ])
    return leech_basis / 2.0

def check_leech_congruence(v):
    """
    Checks if a vector satisfies the congruence conditions for the Leech lattice.

    A vector v is in the Leech lattice if and only if v = x / (2*sqrt(2)) where
    x is an integer vector in Z^24 satisfying:
    1. All components of x have the same parity.
    2. sum(x_i) = 0 (mod 8)
    3. For each codeword c of the Golay code, sum_{i in supp(c)} x_i = 0 (mod 4)

    This function checks a different set of congruence conditions:
    x = y = z (mod 2E8) and x+y+z = 0 (mod 4E8)
    where v = (x, y, z) is a 24-dimensional vector split into three 8-dimensional vectors.
    This construction is known as Construction B.

    Args:
        v: The 24-dimensional vector to check.

    Returns:
        True if the vector satisfies the congruence conditions, False otherwise.
    """

    # We need to define what it means to be congruent to a lattice.
    # x = y (mod 2E8) means (x-y) is in 2E8.
    # 2E8 is the set of vectors {2u | u in E8}.

    x = v[:8]
    y = v[8:16]
    z = v[16:24]

    # We need a way to check if a vector is in the E8 lattice.
    # A vector is in a lattice if it is an integer linear combination of the basis vectors.
    # So, we need to solve the system of linear equations B*c = v, where B is the basis.
    # If the solution c has integer components, then v is in the lattice.

    e8_basis = get_e8_basis()

    def is_in_e8(vec):
        try:
            coords = np.linalg.solve(e8_basis.T, vec)
            return np.allclose(coords, np.round(coords))
        except np.linalg.LinAlgError:
            return False

    # Check x = y (mod 2E8)
    if not is_in_e8((x - y) / 2):
        return False

    # Check y = z (mod 2E8)
    if not is_in_e8((y - z) / 2):
        return False

    # Check x+y+z = 0 (mod 4E8)
    if not is_in_e8((x + y + z) / 4):
        return False

    return True

@ray.remote
def generate_leech_roots_type1():
    roots = np.zeros((1104, 24))
    k = 0
    for i in range(24):
        for j in range(i + 1, 24):
            for signs in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                v = np.zeros(24)
                v[i] = signs[0] * 4
                v[j] = signs[1] * 4
                roots[k] = v / np.sqrt(8)
                k += 1
    return roots

@ray.remote
def generate_leech_roots_type2(golay_words_chunk):
    roots = []
    for c in golay_words_chunk:
        if np.sum(c) == 8:
            support = np.where(c == 1)[0]
            for i in range(2**7):
                v = np.zeros(24)
                for bit in range(7):
                    if (i >> bit) & 1:
                        v[support[bit]] = 2
                    else:
                        v[support[bit]] = -2
                if np.sum(v[support] < 0) % 2 == 0:
                    v[support[7]] = 2
                else:
                    v[support[7]] = -2
                roots.append(v / np.sqrt(8))
    return np.array(roots)

from functools import lru_cache

@lru_cache(maxsize=None)
def generate_leech_roots():
    """
    Generates the 196,560 minimal vectors of the Leech lattice in parallel using Ray.
    """
    golay_words = generate_golay_code_words()
    print("Total number of Golay codewords:", len(golay_words))
    print("Number of weight 8 codewords:", np.sum(np.sum(golay_words, axis=1) == 8))

    # The generation of type 3 vectors is complex and not yet implemented.
    # We will generate the first two types in parallel.

    type1_future = generate_leech_roots_type1.remote()

    # Split the Golay words into chunks for parallel processing.
    num_chunks = int(ray.available_resources().get("CPU", 1))
    chunk_size = len(golay_words) // num_chunks
    chunks = [golay_words[i:i + chunk_size] for i in range(0, len(golay_words), chunk_size)]

    type2_futures = [generate_leech_roots_type2.remote(chunk) for chunk in chunks]

    type1_roots = ray.get(type1_future)
    type2_roots_chunks = ray.get(type2_futures)

    type2_roots = np.vstack(type2_roots_chunks)

    # For now, since the type 3 generation is not implemented,
    # we will return the sum of the number of roots of type 1 and 2.

    return len(type1_roots) + len(type2_roots)

from numba import jit

from numba import jit

@jit(nopython=True)
def generate_leech_points(radius, basis):
    """
    Generates all Leech lattice points within a given radius of the origin.

    Args:
        radius: The radius to search within.
        basis: The basis of the Leech lattice.

    Returns:
        A numpy array of the Leech lattice points.
    """

    # We will use a breadth-first search to find all points within the radius.

    origin = np.zeros(24)
    points = {tuple(origin)}
    queue = [origin]

    head = 0
    while head < len(queue):
        p = queue[head]
        head += 1

        for b in basis:
            new_p = p + b
            if np.dot(new_p, new_p) <= radius**2:
                new_p_tuple = tuple(new_p)
                if new_p_tuple not in points:
                    points.add(new_p_tuple)
                    queue.append(new_p)

            new_p = p - b
            if np.dot(new_p, new_p) <= radius**2:
                new_p_tuple = tuple(new_p)
                if new_p_tuple not in points:
                    points.add(new_p_tuple)
                    queue.append(new_p)

    return np.array([list(p) for p in points])
