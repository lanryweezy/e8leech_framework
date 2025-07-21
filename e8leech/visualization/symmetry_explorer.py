import numpy as np
import matplotlib.pyplot as plt
from e8leech.core.golay_code import E8Lattice, LeechLattice

class SymmetryExplorer:
    """
    A class for exploring the symmetries of the E8 and Leech lattices.
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

    def get_automorphism_group_order(self):
        """
        Returns the order of the automorphism group of the lattice.
        """
        if self.dimension == 8:
            # Order of the Weyl group of E8
            return 696729600
        elif self.dimension == 24:
            # Order of the Conway group Co1
            return 8315553613086720000

    def visualize_root_system(self):
        """
        Visualizes the root system of the E8 lattice.
        """
        if self.dimension != 8:
            raise ValueError("Root system visualization is only supported for the E8 lattice.")

        roots = self.lattice.root_system

        # Project the 8D roots to 2D for visualization
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        roots_2d = pca.fit_transform(roots)

        plt.figure(figsize=(8, 8))
        plt.scatter(roots_2d[:, 0], roots_2d[:, 1], s=10)
        plt.title('2D Projection of the E8 Root System')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.grid(True)
        plt.show()
