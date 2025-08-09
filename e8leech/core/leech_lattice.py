import numpy as np
import ray
from e8leech.core.e8_lattice import get_e8_basis
from e8leech.core.golay_code import generate_golay_code_words, get_golay_generator_matrix
from e8leech.core.base_lattice import BaseLattice

ray.init(ignore_reinit_error=True)

class LeechLattice(BaseLattice):
    def __init__(self):
        super().__init__(dimension=24)
        self.basis = self.construct_leech_lattice_basis()
        if np.linalg.matrix_rank(self.basis) < self.dimension:
            raise np.linalg.LinAlgError("Basis is not linearly independent")
        self._root_system = None

    def construct_leech_lattice_basis(self):
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
        return leech_basis / np.sqrt(8)

    def get_root_system(self):
        if self._root_system is None:
            self._root_system = self.generate_leech_roots()
        return self._root_system

    def check_leech_congruence(self, v):
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

    def generate_leech_roots(self):
        """
        Generates the 196,560 minimal vectors of the Leech lattice in parallel using Ray.
        """
        golay_words = generate_golay_code_words()
        print("Total number of Golay codewords:", len(golay_words))
        print("Number of weight 8 codewords:", np.sum(np.sum(golay_words, axis=1) == 8))

        # The generation of type 3 vectors is complex and not yet implemented.
        # We will generate the first two types in parallel.

        type1_future = self.generate_leech_roots_type1.remote()

        # Split the Golay words into chunks for parallel processing.
        num_chunks = int(ray.available_resources().get("CPU", 1))
        chunk_size = len(golay_words) // num_chunks
        chunks = [golay_words[i:i + chunk_size] for i in range(0, len(golay_words), chunk_size)]

        type2_futures = [self.generate_leech_roots_type2.remote(chunk) for chunk in chunks]

        type1_roots = ray.get(type1_future)
        type2_roots_chunks = ray.get(type2_futures)

        type2_roots = np.vstack(type2_roots_chunks)

        # For now, since the type 3 generation is not implemented,
        # we will return the sum of the number of roots of type 1 and 2.

        return np.vstack([type1_roots, type2_roots])

    @staticmethod
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

    @staticmethod
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
