import numpy as np
from e8leech.core.base_lattice import BaseLattice
from e8leech.core.golay_code import get_golay_generator_matrix

def get_coxeter_todd_basis():
    """
    Returns the basis for the Coxeter-Todd lattice.
    """
    G = get_golay_generator_matrix()
    I = np.identity(12)

    A = np.hstack([G, np.zeros((12, 12))])
    B = np.hstack([np.zeros((12, 12)), G])

    C = np.hstack([I, I])
    D = np.hstack([I, -I])

    M = np.vstack([
        np.hstack([A, B]),
        np.hstack([C, D])
    ])

    return M[:12, :12]

class CoxeterToddLattice(BaseLattice):
    def __init__(self):
        super().__init__(dimension=12)
        self.basis = get_coxeter_todd_basis()
