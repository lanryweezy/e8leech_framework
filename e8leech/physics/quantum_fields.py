import numpy as np

class QuantumField:
    """
    A quantum field on a lattice.
    """
    def __init__(self, lattice_vectors):
        """
        Initializes the quantum field.

        Args:
            lattice_vectors: The vectors of the lattice.
        """
        self.lattice_vectors = lattice_vectors
        self.num_sites = len(lattice_vectors)
        self.field = np.zeros(self.num_sites, dtype=np.complex128)

    def get_operator(self, site_index):
        """
        Returns the field operator for a given site.
        """
        op = np.zeros((self.num_sites, self.num_sites), dtype=np.complex128)
        op[site_index, site_index] = 1
        return op

    def get_creation_operator(self, site_index):
        """
        Returns the creation operator for a given site.
        """
        # In a simple model, the creation operator is just the field operator.
        return self.get_operator(site_index)

    def get_annihilation_operator(self, site_index):
        """
        Returns the annihilation operator for a given site.
        """
        # In a simple model, the annihilation operator is the adjoint of the field operator.
        return self.get_operator(site_index).T.conj()
