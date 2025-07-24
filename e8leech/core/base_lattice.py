import numpy as np

class BaseLattice:
    def __init__(self, dimension):
        self.dimension = dimension
        self.basis = None
        self.root_system = None

    def lll_reduction(self):
        """
        Performs LLL basis reduction on the lattice basis.
        """
        if self.basis is None:
            return

        from fpylll import IntegerMatrix, LLL

        if np.linalg.matrix_rank(self.basis) < self.dimension:
            raise np.linalg.LinAlgError("Basis is not linearly independent")

        M = IntegerMatrix.from_matrix(self.basis.astype(int).tolist())
        reduced_basis = LLL.reduction(M)
        return np.array([list(row) for row in reduced_basis])

    def is_valid(self, v):
        """
        Checks if a vector is a valid point in the lattice.
        A vector is in the lattice if it is an integer linear combination of the basis vectors.
        """
        if self.basis is None:
            return False
        try:
            coords = np.linalg.solve(self.basis.T, v)
            return np.allclose(coords, np.round(coords))
        except np.linalg.LinAlgError:
            return False

    def is_even(self, v):
        """
        Checks if a vector has an even squared norm.
        """
        return np.isclose(np.dot(v, v) % 2, 0) or np.isclose(np.dot(v, v) % 2, 2)

    def cvp(self, v):
        """
        Solves the closest vector problem (CVP) for the given vector.
        """
        if self.basis is None:
            return None

        from e8leech.core.cvp import kannan_fincke_pohst
        return kannan_fincke_pohst(v, self.basis)

    def quantize(self, v):
        """
        Quantizes a vector to the nearest lattice point.
        This is equivalent to the closest vector problem (CVP).
        """
        return self.cvp(v)

    def babai_nearest_plane(self, v):
        """
        Babai's nearest plane algorithm for solving the closest vector problem.
        """
        if self.basis is None:
            return None

        reduced_basis = self.lll_reduction()

        if reduced_basis.ndim == 0:
            reduced_basis = np.array([reduced_basis])

        b = reduced_basis.T
        g = np.dot(b, b.T) # Gram matrix
        g_inv = np.linalg.inv(g)

        y = np.dot(v, np.dot(b.T, g_inv))
        c = np.round(y).astype(int)

        return np.dot(c, b)
