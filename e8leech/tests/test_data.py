import unittest
import numpy as np
from e8leech.data.quantization import quantize_to_e8, quantize_to_leech
from e8leech.core.e8_lattice import get_e8_basis
from e8leech.core.leech_lattice import construct_leech_lattice_basis
from e8leech.data.lsh import LSH

class TestData(unittest.TestCase):

    def test_quantize_to_e8(self):
        """
        Tests the quantization to the E8 lattice.
        """
        vectors = np.random.rand(10, 8)
        q_vectors = quantize_to_e8(vectors)
        # The quantized vectors should be points in the E8 lattice.
        # We can check this by verifying that their coordinates in the E8 basis are integers.
        basis = get_e8_basis()
        for q_v in q_vectors:
            coords = np.dot(q_v, np.linalg.inv(basis))
            self.assertTrue(np.allclose(coords, np.round(coords)))

    def test_quantize_to_leech(self):
        """
        Tests the quantization to the Leech lattice.
        """
        vectors = np.random.rand(10, 24)
        q_vectors = quantize_to_leech(vectors)
        # The quantized vectors should be points in the Leech lattice.
        # We can check this by verifying that their coordinates in the Leech basis are integers.
        basis = construct_leech_lattice_basis()
        for q_v in q_vectors:
            coords = np.dot(q_v, np.linalg.inv(basis))
            self.assertTrue(np.allclose(coords, np.round(coords)))

    def test_lsh(self):
        """
        Tests the LSH implementation.
        """
        data = np.random.rand(100, 8)
        lsh = LSH(num_hashes=10, input_dim=8, bucket_width=0.5)
        lsh.index(data)

        query_vectors = data[:10]
        neighbors = lsh.query(query_vectors)

        # The first neighbor of each query vector should be the query vector itself.
        for i, n in enumerate(neighbors):
            self.assertIn(i, n)

if __name__ == '__main__':
    unittest.main()
