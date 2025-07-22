import numpy as np

def project_to_subspace(data, num_dims):
    """
    Projects data onto a lower-dimensional subspace using PCA.

    Args:
        data: The data to be projected.
        num_dims: The number of dimensions to project onto.

    Returns:
        The projected data.
    """

    # This is a standard dimension reduction technique.
    # It is not specific to lattices, but it can be used in conjunction with them.

    from sklearn.decomposition import PCA

    pca = PCA(n_components=num_dims)
    projected_data = pca.fit_transform(data)

    return projected_data
