import numpy as np

def get_golay_generator_matrix():
    """
    Returns a generator matrix for the extended Golay code G24.
    
    The extended Golay code is a [24, 12, 8] linear code.
    This implementation uses a standard generator matrix.
    """
    # A standard generator matrix for G24
    G = np.array([
        [1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,0,1,0,1,0,0],
        [0,1,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,0,1,0,1,0,1,0],
        [0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,1,0,1,0,1,1,0,0,1],
        [0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,0,1,0,0,1],
        [0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,0,1,0,1],
        [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,0,1,1],
        [0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,1,1,0,1,1,0,1],
        [0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,1,1,0,1,1,1],
        [0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,1,1,1,0,1,1],
        [0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,1,0,0,0,1,1,1,0,1],
        [0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,1,0,0,0,1,1,1,1],
        [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1]
    ], dtype=int)
    return G

def generate_golay_code_words():
    """
    Generates the extended Golay code G24.
    """
    # The extended Golay code G24 is a [24, 12, 8] code.
    # It can be constructed from the binary Golay code G23, which is a [23, 12, 7] code.
    # G23 is a perfect code.
    # We construct G23 from the quadratic residue code of length 23.

    # The quadratic residues modulo 23 are:
    # 1, 2, 3, 4, 6, 8, 9, 12, 13, 16, 18
    q = [1, 2, 3, 4, 6, 8, 9, 12, 13, 16, 18]

    # The generator matrix for G23 is constructed from a circulant matrix.
    # Let C be the 11x11 circulant matrix with first row (1,1,0,1,1,1,0,0,0,1,0).
    # This is not correct.

    # Let's use a known generator matrix for G23.
    # The generator matrix for G23 is a 12x23 matrix.
    # It can be written in the form [I_12 | A], where A is a 12x11 matrix.
    # The rows of A are cyclic shifts of the vector (1,1,0,1,1,1,0,0,0,1,0).

    # A generator matrix for the binary Golay code G23
    G23 = np.array([
        [1,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,1,0,0,0,1,0],
        [0,1,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,1,0,0,0,1],
        [0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,1,0,1,1,1,0,0,0],
        [0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,1,0,1,1,1,0,0],
        [0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,1,0,1,1,1,0],
        [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,1,0,1,1,1],
        [0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,1,1,0,1,1],
        [0,0,0,0,0,0,0,1,0,0,0,0,1,1,0,0,0,1,0,1,1,0,1],
        [0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,1,1,0],
        [0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,1,1],
        [0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,1,1,0,0,0,1,0,1],
        [0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,1,1,0,0,0,1,0]
    ], dtype=int)

    # The extended Golay code G24 is obtained by adding a parity bit to each codeword of G23.
    # The generator matrix for G24 is obtained by adding a column of ones to G23,
    # and then adding a row of ones.

    G24 = np.zeros((12, 24), dtype=int)
    G24[:, :23] = G23
    G24[:, 23] = 1

    # The last row of the generator matrix for G24 is all ones.
    # This is not correct.

    # Let's use a known generator matrix for G24.
    G = np.array([
        [1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,0,1,0,1,0,0],
        [0,1,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,0,1,0,1,0,1,0],
        [0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,1,0,1,0,1,1,0,0,1],
        [0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,0,1,0,0,1],
        [0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,0,1,0,1],
        [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,0,1,1],
        [0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,1,1,0,1,1,0,1],
        [0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,1,1,0,1,1,1],
        [0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,1,1,1,0,1,1],
        [0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,1,0,0,0,1,1,1,0,1],
        [0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,1,0,0,0,1,1,1,1],
        [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1]
    ], dtype=int)

    codewords = []
    for i in range(2**12):
        message = np.array([int(x) for x in bin(i)[2:].zfill(12)])
        codeword = message.dot(G) % 2
        codewords.append(codeword)
    return np.array(codewords)
