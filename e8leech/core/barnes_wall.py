import numpy as np
from e8leech.core.base_lattice import BaseLattice

def get_d4_basis():
    """
    Returns the basis for the D4 lattice.
    """
    return np.array([
        [1, 1, 0, 0],
        [-1, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 1]
    ])

def get_barnes_wall_basis(n):
    """
    Returns the basis for the Barnes-Wall lattice of dimension 2^n.
    """
    if n == 1:
        return get_d4_basis()

    basis_prev = get_barnes_wall_basis(n - 1)
    dim_prev = basis_prev.shape[0]

    A = np.hstack([basis_prev, basis_prev])
    B = np.hstack([np.zeros((dim_prev, dim_prev)), 2 * np.identity(dim_prev)])

    return np.vstack([A, B])

class BarnesWallLattice(BaseLattice):
    def __init__(self, n):
        self.n = n
        dimension = 2**n
        super().__init__(dimension)
        self.basis = get_barnes_wall_basis(n)
