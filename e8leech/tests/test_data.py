import unittest
import numpy as np
from e8leech.data.quantization import quantize_to_e8, quantize_to_leech
from e8leech.core.e8_lattice import get_e8_basis
from e8leech.core.leech_lattice import construct_leech_lattice_basis

class TestQuantization(unittest.TestCase):

    def test_quantize_to_e8(self):
        """
        Tests the quantization to the E8 lattice.
        """
        v = np.random.rand(8)
        q_v = quantize_to_e8(v)
        # The quantized vector should be a point in the E8 lattice.
        # We can check this by verifying that its coordinates in the E8 basis are integers.
        basis = get_e8_basis()
        coords = np.dot(q_v, np.linalg.inv(basis))
        self.assertTrue(np.allclose(coords, np.round(coords)))

    def test_quantize_to_leech(self):
        """
        Tests the quantization to the Leech lattice.
        """
        v = np.random.rand(24)
        q_v = quantize_to_leech(v)
        # The quantized vector should be a point in the Leech lattice.
        # We can check this by verifying that its coordinates in the Leech basis are integers.
        basis = construct_leech_lattice_basis()
        coords = np.dot(q_v, np.linalg.inv(basis))
        self.assertTrue(np.allclose(coords, np.round(coords)))

if __name__ == '__main__':
    unittest.main()
