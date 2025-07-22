import unittest
import numpy as np
from hypothesis import given
from hypothesis.extra.numpy import arrays
from e8leech.core.e8_lattice import generate_e8_roots, get_e8_basis, babai_nearest_plane

class TestE8Lattice(unittest.TestCase):

    def test_e8_root_system_size(self):
        """
        Tests that the E8 root system has exactly 240 vectors.
        """
        roots = generate_e8_roots()
        self.assertEqual(roots.shape[0], 240)

    def test_e8_root_system_norms(self):
        """
        Tests that all root vectors in the E8 lattice have a squared norm of 2.
        """
        roots = generate_e8_roots()
        for v in roots:
            self.assertAlmostEqual(np.dot(v, v), 2.0)

    def test_e8_basis(self):
        """
        Tests that the E8 basis vectors are linearly independent.
        """
        basis = get_e8_basis()
        self.assertEqual(np.linalg.matrix_rank(basis), 8)

    def test_babai_nearest_plane(self):
        """
        Tests Babai's nearest plane algorithm for the E8 lattice.
        """
        basis = get_e8_basis()
        v = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        closest_point = babai_nearest_plane(v, basis)
        # The closest point to v should be a lattice point.
        # We can check this by verifying that its coordinates in the basis are integers.
        coords = np.dot(closest_point, np.linalg.inv(basis))
        self.assertTrue(np.allclose(coords, np.round(coords)))

    def test_e8_packing_density(self):
        """
        Tests that the E8 lattice has a packing density of approx 0.25367.
        """
        # The packing density of a lattice is given by the formula:
        # delta = (volume of one sphere) / (volume of fundamental parallelepiped)
        # The radius of the spheres is half the minimum distance between lattice points.
        # For E8, the minimum squared norm is 2, so the minimum distance is sqrt(2).
        # The radius is sqrt(2)/2.
        # The volume of an 8D sphere is (pi^4 / 24) * r^8
        # The volume of the fundamental parallelepiped is det(B), where B is the basis matrix.

        radius = np.sqrt(2) / 2
        volume_sphere = (np.pi**4 / 24) * (radius**8)

        basis = get_e8_basis()
        volume_parallelepiped = np.linalg.det(basis)

        packing_density = volume_sphere / volume_parallelepiped
        self.assertAlmostEqual(packing_density, 0.25367, places=5)

from hypothesis import settings

    @settings(max_examples=10)
    @given(arrays(np.float64, (8,)))
    def test_babai_property(self, v):
        """
        Tests a property of Babai's nearest plane algorithm.
        The distance between a vector and its quantization should be less than
        the distance between the vector and any other lattice point.
        """
        basis = get_e8_basis()
        q_v = babai_nearest_plane(v, basis)

        # Generate another lattice point.
        c = np.random.randint(-10, 10, 8)
        other_lattice_point = np.dot(c, basis)

        if not np.allclose(q_v, other_lattice_point):
            dist_to_quantized = np.linalg.norm(v - q_v)
            dist_to_other = np.linalg.norm(v - other_lattice_point)
            self.assertLessEqual(dist_to_quantized, dist_to_other)

    def test_even_lattice(self):
        """
        Tests that the E8 lattice is an even lattice.
        """
        basis = get_e8_basis()
        # Check that the squared norm of each basis vector is an even integer.
        for v in basis:
            self.assertEqual(int(np.dot(v, v)) % 2, 0)

        # Check that the dot product of any two basis vectors is an integer.
        for i in range(len(basis)):
            for j in range(i, len(basis)):
                self.assertAlmostEqual(np.dot(basis[i], basis[j]) % 1, 0)

if __name__ == '__main__':
    unittest.main()
