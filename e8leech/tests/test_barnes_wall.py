import numpy as np
from e8leech.core.barnes_wall import BarnesWallLattice, get_d4_basis

def test_barnes_wall_basis_n1():
    bw1 = BarnesWallLattice(1)
    d4 = get_d4_basis()
    assert np.allclose(bw1.basis, d4)

def test_barnes_wall_basis_n2():
    bw2 = BarnesWallLattice(2)
    assert bw2.basis.shape == (8, 8)
