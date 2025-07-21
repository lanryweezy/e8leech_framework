import numpy as np

def compactify_on_torus(lattice_basis, dimensions):
    """
    Performs a string compactification on a torus defined by a lattice.

    Args:
        lattice_basis: The basis of the lattice defining the torus.
        dimensions: The number of dimensions to compactify.

    Returns:
        The compactified lattice basis.
    """

    # This is a simplified model of string compactification.
    # A more realistic model would involve Calabi-Yau manifolds.
    # For now, we will just consider a toroidal compactification.

    # The compactified dimensions are those that are "curled up" into a torus.
    # The size and shape of the torus are determined by the lattice.
    # The compactified lattice basis is obtained by projecting the original basis
    # onto the remaining non-compact dimensions.

    num_dims = lattice_basis.shape[1]
    if dimensions > num_dims:
        raise ValueError("Number of dimensions to compactify cannot be greater than the total number of dimensions.")

    # The non-compact dimensions are the first (num_dims - dimensions) dimensions.
    non_compact_dims = num_dims - dimensions

    # The compactified basis is the projection of the original basis onto the non-compact dimensions.
    # This is equivalent to taking the first `non_compact_dims` columns of the basis.
    compactified_basis = lattice_basis[:, :non_compact_dims]

    return compactified_basis
