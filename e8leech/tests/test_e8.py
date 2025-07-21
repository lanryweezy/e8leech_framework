import unittest
import numpy as np
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

if __name__ == '__main__':
    unittest.main()
