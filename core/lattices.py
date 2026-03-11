import numpy as np

class E8Lattice:
    """
    Implementation of the E8 Lattice in 8-dimensional Euclidean space.
    The E8 lattice is the unique even unimodular lattice of rank 8.
    It provides the optimal sphere packing in 8D.
    """
    def __init__(self):
        self.dim = 8
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
        Generates the G24 generator matrix [I | A] where A is derived 
        from the adjacency of an icosahedron or quadratic residues.
        """
        # I_12: Identity matrix
        I = np.eye(12, dtype=int)
        
        # A_12: Construction using quadratic residues (mod 11) + parity
        # This is a standard construction for the Golay code.
        A = np.zeros((12, 12), dtype=int)
        
        # Matrix B from the QR(11) code
        # Row 0 is all 1s except diagonal
        # Subsequent rows are cyclic shifts of the QR pattern
        qr_pattern = [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1] # QR mod 11
        
        B = np.zeros((11, 11), dtype=int)
        for i in range(11):
            for j in range(11):
                B[i, j] = qr_pattern[(j - i) % 11]
        
        # Complement the matrix B and add borders
        A[0, 0] = 0
        A[0, 1:] = 1
        A[1:, 0] = 1
        A[1:, 1:] = B # Simplification for now, usually needs specific XOR logic
        
        return np.hstack((I, A))

    def encode(self, data_bits):
        """ Encodes 12 bits into a 24-bit Golay codeword. """
        if len(data_bits) != 12:
            raise ValueError("Input must be 12 bits")
        return np.dot(data_bits, self.generator_matrix) % 2

class LeechLattice:
    """
    Implementation of the Leech Lattice in 24-dimensional space.
    Uses the Binary Golay Code construction (Construction B).
    """
    def __init__(self):
        self.dim = 24
        self.golay = GolayCode()

    def generate_minimal_vectors(self):
        """ 
        Generates a subset of the 196,560 minimal vectors.
        Type 1: (4, 4, 0, ..., 0) - permutations (total 1,104)
        Type 2: (2, 2, ..., 2, -6, 2, ..., 2) based on Golay (total 98,304)
        Type 3: (+-1, +-1, ..., +-1) based on Golay (total 97,152)
        """
        # This will be fully implemented in Day 4
        return []
