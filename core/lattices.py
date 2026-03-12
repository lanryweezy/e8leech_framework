import numpy as np

class Lattice:
    """ Base class for lattices. """
    def __init__(self, dim):
        self.dim = dim

    def apply_symmetry(self, vector, permutation=None, signs=None):
        """ 
        Applies basic symmetry operations: coordinate permutation and sign flips.
        """
        v = np.copy(vector)
        if signs is not None:
            v = v * signs
        if permutation is not None:
            v = v[permutation]
        return v

class E8Lattice(Lattice):
    """
    Implementation of the E8 Lattice in 8-dimensional Euclidean space.
    The E8 lattice is the unique even unimodular lattice of rank 8.
    It provides the optimal sphere packing in 8D.
    """
    def __init__(self):
        super().__init__(8)
        self.basis = self._generate_basis()

    def _generate_basis(self):
        """
        Generates an upper triangular basis for the E8 lattice.
        Coordinates are either all integers or all half-integers, 
        and the sum of coordinates is even.
        """
        # Basis matrix where columns generate the lattice
        B = np.zeros((8, 8))
        for i in range(6):
            B[i, i] = 1
            B[i+1, i] = -1
        B[6, 6] = 1
        B[7, 6] = 1
        
        # The 8th vector involves the half-integers
        B[:, 7] = -0.5 * np.ones(8)
        
        return B

    def is_in_lattice(self, vector):
        """ Checks if a given 8D vector belongs to the E8 lattice. """
        if len(vector) != 8:
            return False
        
        # Check if all are integers
        all_int = np.all(np.mod(vector, 1) == 0)
        # Check if all are half-integers
        all_half = np.all(np.mod(vector, 1) == 0.5)
        
        if not (all_int or all_half):
            return False
            
        # Sum must be even
        if np.sum(vector) % 2 != 0:
            return False
            
        return True

    def quantify(self, x):
        """ 
        Finds the closest point in the E8 lattice to an arbitrary 8D vector x.
        This is the "decoding" algorithm for E8.
        """
        # Algorithm by Conway and Sloane
        
        # 1. Round to nearest integer vector f(x)
        f_x = np.round(x)
        if np.sum(f_x) % 2 != 0:
            # Find coordinate with largest rounding error and flip it
            errors = np.abs(x - f_x)
            k = np.argmax(errors)
            f_x[k] += 1 if x[k] > f_x[k] else -1
            
        # 2. Round to nearest half-integer vector g(x)
        # g(x) = f(x - 1/2) + 1/2
        g_x_shifted = np.round(x - 0.5)
        if (np.sum(g_x_shifted) + 4) % 2 != 0: # +4 to handle parity of 0.5*8
            errors = np.abs((x - 0.5) - g_x_shifted)
            k = np.argmax(errors)
            g_x_shifted[k] += 1 if (x[k] - 0.5) > g_x_shifted[k] else -1
        g_x = g_x_shifted + 0.5
        
        # 3. Compare distances
        dist_f = np.linalg.norm(x - f_x)
        dist_g = np.linalg.norm(x - g_x)
        
        return f_x if dist_f < dist_g else g_x

    def get_shortest_vectors(self):
        """ 
        Returns the 240 roots (shortest non-zero vectors) of E8.
        These have a norm squared of 2 (length sqrt(2)).
        """
        roots = []
        # Type 1: (+-1, +-1, 0, 0, 0, 0, 0, 0) - permutations (112 vectors)
        for i in range(8):
            for j in range(i + 1, 8):
                for s1 in [1, -1]:
                    for s2 in [1, -1]:
                        v = np.zeros(8)
                        v[i], v[j] = s1, s2
                        roots.append(v)
        
        # Type 2: (+-1/2, +-1/2, ..., +-1/2) with even number of minus signs (128 vectors)
        for i in range(256):
            bits = [(i >> j) & 1 for j in range(8)]
            if sum(bits) % 2 == 0: 
                v = np.array([0.5 if b == 0 else -0.5 for b in bits])
                roots.append(v)
                
        return np.array(roots)

class GolayCode:
    """
    Implementation of the Extended Binary Golay Code [24, 12, 8].
    This is used as a foundation for constructing the Leech Lattice.
    """
    def __init__(self):
        self.generator_matrix = self._generate_g24()

    def _generate_g24(self):
        """ 
        Generates the G24 generator matrix [I | A] where A is the adjacency 
        of the quadratic residue code mod 11.
        """
        I = np.eye(12, dtype=int)
        
        # QR mod 11: 1, 3, 4, 5, 9
        res = [1, 3, 4, 5, 9]
        
        # B is the 11x11 circulant matrix for the QR code
        B = np.zeros((11, 11), dtype=int)
        for i in range(11):
            for j in range(11):
                if (j - i) % 11 in res:
                    B[i, j] = 1
        
        # A is 12x12
        A = np.zeros((12, 12), dtype=int)
        A[0, 0] = 0
        A[0, 1:] = 1
        A[1:, 0] = 1
        # The remainder is J - B - I mod 2 (which is 1 - B - Identity)
        # Actually, for the extended Golay code, A[1:, 1:] is usually B with some modification.
        # Let's use the known symmetry: A is the adjacency matrix of the icosahedron 
        # but for G24 it's more specific.
        
        # Correct construction: A[1:, 1:] = B + B^T + I? No.
        # Let's try: A[1:, 1:] = B (if B is the QR matrix)
        # And we need to ensure the weights are correct.
        A[1:, 1:] = B
        # Flip diagonal? No.
        
        # Let's try the "Standard" A matrix for G24:
        # 0 1 1 1 1 1 1 1 1 1 1 1
        # 1 1 1 0 1 1 1 0 0 0 1 0
        # 1 0 1 1 0 1 1 1 0 0 0 1
        # ... (cyclic shifts of 1 1 0 1 1 1 0 0 0 1 0)
        
        row1 = [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0]
        for i in range(11):
            for j in range(11):
                A[i+1, j+1] = row1[(j-i)%11]
        
        return np.hstack((I, A))

    def encode(self, data_bits):
        """ Encodes 12 bits into a 24-bit Golay codeword. """
        if len(data_bits) != 12:
            raise ValueError("Input must be 12 bits")
        return np.dot(data_bits, self.generator_matrix) % 2

    def get_all_codewords(self):
        """ Generates all 4096 codewords of the [24, 12, 8] code. """
        codewords = []
        for i in range(4096):
            bits = np.array([(i >> j) & 1 for j in range(12)])
            codewords.append(self.encode(bits))
        return np.array(codewords)

class LeechLattice(Lattice):
    """
    Implementation of the Leech Lattice in 24-dimensional space.
    Uses the Binary Golay Code construction (Construction B).
    """
    def __init__(self):
        super().__init__(24)
        self.golay = GolayCode()

    def generate_minimal_vectors(self):
        """ 
        Generates the 196,560 minimal vectors of norm 4.
        """
        minimal_vectors = []
        
        # Shape 1: (4, 4, 0^22) -> 1,104 vectors
        for i in range(24):
            for j in range(i + 1, 24):
                for s1 in [4, -4]:
                    for s2 in [4, -4]:
                        v = np.zeros(24)
                        v[i], v[j] = s1, s2
                        minimal_vectors.append(v)
        
        # Get codewords for Shape 2 and 3
        codewords = self.golay.get_all_codewords()
        
        # Shape 2: (2^8, 0^16) -> 97,152 vectors
        # These are based on octads (weight 8 codewords)
        octads = codewords[np.sum(codewords, axis=1) == 8]
        for octad in octads:
            indices = np.where(octad == 1)[0]
            # There are 2^7 sign combinations such that product of signs is 1 (even sum)
            for i in range(256):
                signs = np.array([1 if (i >> j) & 1 == 0 else -1 for j in range(8)])
                # Sum of indices where sign is -1 must be even
                if np.sum(signs == -1) % 2 == 0:
                    v = np.zeros(24)
                    v[indices] = 2 * signs
                    minimal_vectors.append(v)

        # Shape 3: (3, 1^23) -> 98,304 vectors
        # For each coordinate i, there is a vector v where v_i = -3 (or 3)
        # and v_j = 1 or -1. The pattern of signs must correspond to a codeword.
        for i in range(24):
            for codeword in codewords:
                v = np.array([1 if b == 0 else -1 for b in codeword], dtype=float)
                # If we want v[i] to be -3, we need to ensure the original v[i] was 1.
                # If original v[i] was -1, then v[i] becomes 3.
                # v[i] = v[i] - 4 if codeword[i] == 0 else v[i] + 4
                if codeword[i] == 0:
                    v[i] = -3
                else:
                    v[i] = 3
                minimal_vectors.append(v)
                
        return np.array(minimal_vectors)
