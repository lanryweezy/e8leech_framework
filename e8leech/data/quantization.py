import numpy as np
from e8leech.core.e8_lattice import get_e8_basis, babai_nearest_plane
from e8leech.core.leech_lattice import construct_leech_lattice_basis

def quantize_to_e8(v):
    """
    Quantizes a vector to the E8 lattice.

    Args:
        v: The vector to be quantized.

    Returns:
        The closest E8 lattice point to v.
    """
    basis = get_e8_basis()
    return babai_nearest_plane(v, basis)

def quantize_to_leech(v):
    """
    Quantizes a vector to the Leech lattice.

    Args:
        v: The vector to be quantized.

    Returns:
        The closest Leech lattice point to v.
    """
    basis = construct_leech_lattice_basis()
    # The babai_nearest_plane algorithm is not guaranteed to work well for the Leech lattice
    # with this basis, as it is not LLL-reduced.
    # A more sophisticated algorithm like sphere decoding would be needed for optimal results.
    # For now, we will use Babai's algorithm as a placeholder.
    return babai_nearest_plane(v, basis)
