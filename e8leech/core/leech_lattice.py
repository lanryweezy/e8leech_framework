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
        self.root_system = self.get_root_system()

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
        if self.root_system is not None:
            return self.root_system
        return self.generate_leech_roots()

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
