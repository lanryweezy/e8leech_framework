import numpy as np
from e8leech.core.e8_lattice import get_e8_basis, babai_nearest_plane
from e8leech.core.leech_lattice import LeechLattice

def quantize_to_e8(vectors):
    """
    Quantizes a batch of vectors to the E8 lattice.

    Args:
        vectors: A numpy array of vectors to be quantized.

    Returns:
        A numpy array of the closest E8 lattice points.
    """
    basis = get_e8_basis()
    return np.array([babai_nearest_plane(v, basis) for v in vectors])

def quantize_to_leech(vectors):
    """
    Quantizes a batch of vectors to the Leech lattice.

    Args:
        vectors: A numpy array of vectors to be quantized.

    Returns:
        A numpy array of the closest Leech lattice points.
    """
    leech = LeechLattice()
    return np.array([leech.quantize(v) for v in vectors])
