import numpy as np
from e8leech.core.golay_code import E8Lattice, LeechLattice, GolayCode

def test_e8_lattice():
    e8 = E8Lattice()
    assert e8.dimension == 8
    assert e8.kissing_number() == 240
    # Test if the origin is on the lattice
    assert e8.is_on_lattice(np.zeros(8))
    # Test a known lattice point
    point = np.array([2, 0, 0, 0, 0, 0, 0, 0])
    assert e8.is_on_lattice(point)

def test_leech_lattice():
    leech = LeechLattice()
    assert leech.dimension == 24
    assert leech.kissing_number() == 196560
    # Test if the origin is on the lattice
    assert leech.is_on_lattice(np.zeros(24))

def test_golay_code():
    golay = GolayCode()
    message = np.random.randint(0, 2, 12)
    codeword = golay.encode(message)
    assert len(codeword) == 24
    decoded_message = golay.decode(codeword)
    assert np.array_equal(message, decoded_message[:12])
