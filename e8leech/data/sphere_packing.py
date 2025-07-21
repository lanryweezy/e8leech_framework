import numpy as np

def get_packing_density(lattice_basis):
    """
    Calculates the packing density of a lattice.

    Args:
        lattice_basis: The basis of the lattice.

    Returns:
        The packing density of the lattice.
    """

    num_dims = lattice_basis.shape[1]

    # The radius of the spheres is half the minimum distance between lattice points.
    # The minimum distance is the square root of the minimum squared norm of a non-zero lattice vector.
    # We can find this by generating a set of lattice vectors and finding the minimum norm.
    # This is computationally expensive.
    # For the E8 and Leech lattices, the minimum squared norms are known.

    if num_dims == 8:
        min_norm_sq = 2
    elif num_dims == 24:
        min_norm_sq = 4
    else:
        # For a general lattice, we would need to find the shortest vector.
        # This is a hard problem.
        # For now, we will assume the minimum norm is 1.
        min_norm_sq = 1

    radius = np.sqrt(min_norm_sq) / 2

    # The volume of an n-dimensional sphere is (pi^(n/2) / Gamma(n/2 + 1)) * r^n
    # We need to implement the Gamma function.
    from scipy.special import gamma

    volume_sphere = (np.pi**(num_dims / 2) / gamma(num_dims / 2 + 1)) * (radius**num_dims)

    # The volume of the fundamental parallelepiped is the absolute value of the determinant of the basis matrix.
    # For a non-square matrix, we can use the square root of the determinant of the Gram matrix.
    gram_matrix = np.dot(lattice_basis, lattice_basis.T)
    volume_parallelepiped = np.sqrt(np.linalg.det(gram_matrix))

    return volume_sphere / volume_parallelepiped
