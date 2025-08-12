import numpy as np
from e8leech.core.coxeter_todd import CoxeterToddLattice

def test_coxeter_todd_basis():
    ct = CoxeterToddLattice()
    assert ct.basis.shape == (12, 12)
