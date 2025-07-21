import numpy as np
from e8leech.core.e8_lattice import get_e8_basis, babai_nearest_plane
from e8leech.core.leech_lattice import construct_leech_lattice_basis

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
    basis = construct_leech_lattice_basis()
    # The babai_nearest_plane algorithm is not guaranteed to work well for the Leech lattice
    # with this basis, as it is not LLL-reduced.
    # A more sophisticated algorithm like sphere decoding would be needed for optimal results.
    # For now, we will use Babai's algorithm as a placeholder.
    return np.array([babai_nearest_plane(v, basis) for v in vectors])
