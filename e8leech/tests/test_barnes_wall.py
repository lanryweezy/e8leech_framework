import numpy as np
from e8leech.core.barnes_wall import BarnesWallLattice, get_d4_basis

def test_barnes_wall_basis():
    bw1 = BarnesWallLattice(1)
    d4 = get_d4_basis()
    assert np.allclose(bw1.basis, d4)
