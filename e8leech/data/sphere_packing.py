from e8leech.core.golay_code import E8Lattice, LeechLattice
import numpy as np

class SpherePacking:
    """
    A class for optimal sphere packing using E8 and Leech lattices.
    """

    def __init__(self, lattice_type='E8'):
        if lattice_type == 'E8':
            self.lattice = E8Lattice()
            self.dimension = 8
        elif lattice_type == 'Leech':
            self.lattice = LeechLattice()
            self.dimension = 24
        else:
            raise ValueError("Unsupported lattice type")

    def get_packing_density(self):
        """
        Calculates the packing density of the lattice.
        """
        # The formula for packing density is:
        # delta = (volume of one sphere) / (volume of fundamental parallelotope)
        # For E8, this is pi^4 / 384
        # For Leech, this is pi^12 / 12! * (479001600)^-1

        if self.dimension == 8:
            return (np.pi**4) / 384
        elif self.dimension == 24:
            # The volume of the fundamental parallelotope of the Leech lattice is 1.
            # The volume of a 24-dimensional sphere is pi^12 / 12! * R^24
            # The radius of the spheres in the Leech lattice packing is 2.
            return (np.pi**12) / np.math.factorial(12) * (2**24) / 1

    def get_kissing_number(self):
        """
        Returns the kissing number of the lattice.
        """
        return self.lattice.kissing_number()
