import numpy as np
from hypothesis import given, strategies as st, settings

from e8leech.core.leech_lattice import LeechLattice

leech = LeechLattice()

@settings(deadline=None)
@given(st.integers(min_value=0, max_value=len(leech.get_root_system()) - 1))
def test_leech_root_system_norm(vector_index):
    """Tests that all root vectors in the Leech lattice have a squared norm of 4."""
    root_system = leech.get_root_system()
    vector = root_system[vector_index]
    assert np.isclose(np.dot(vector, vector), 4)

@settings(deadline=None)
@given(st.integers(min_value=0, max_value=len(leech.get_root_system()) - 1))
def test_leech_is_even(vector_index):
    """Tests that the Leech lattice is an even lattice."""
    root_system = leech.get_root_system()
    vector = root_system[vector_index]
    assert leech.is_even(vector)

@settings(deadline=None)
@given(st.lists(st.floats(min_value=-1, max_value=1), min_size=24, max_size=24))
def test_leech_quantization(vector):
    """Tests the quantization of a vector to the Leech lattice."""
    v = np.array(vector)
    quantized_vector = leech.quantize(v)
    assert np.linalg.norm(v - quantized_vector) <= 2 * np.linalg.norm(v)

@settings(deadline=None)
@given(st.lists(st.floats(min_value=-1, max_value=1), min_size=24, max_size=24))
def test_leech_babai_algorithm(vector):
    """Tests Babai's algorithm for the Leech lattice."""
    v = np.array(vector)
    nearest_vector = leech.babai_nearest_plane(v)
    assert np.linalg.norm(v - nearest_vector) <= 2 * np.linalg.norm(v)
