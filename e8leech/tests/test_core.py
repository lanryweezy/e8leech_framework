import unittest
import numpy as np
from e8leech.core.theta_functions import get_e8_theta_series, get_leech_theta_series

class TestCore(unittest.TestCase):

    def test_e8_theta_series(self):
        """
        Tests the theta series of the E8 lattice.
        """
        theta_series = get_e8_theta_series(num_terms=3)
        # The first few terms are 1 + 240q^2 + 2160q^4 + 6720q^6
        self.assertIn("240*q**2", theta_series)
        self.assertIn("2160*q**4", theta_series)
        self.assertIn("6720*q**6", theta_series)

    def test_leech_theta_series(self):
        """
        Tests the theta series of the Leech lattice.
        """
        theta_series = get_leech_theta_series(num_terms=3)
        # The first few terms are 1 + 196560q^2 + 16773120q^3
        self.assertIn("196560*q**2", theta_series)
        self.assertIn("16773120*q**3", theta_series)

if __name__ == '__main__':
    unittest.main()
