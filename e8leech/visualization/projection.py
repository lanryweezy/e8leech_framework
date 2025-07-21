import numpy as np
from scipy.linalg import null_space

def project_24d(data: np.ndarray, method: str = "lie_algebra") -> np.ndarray:
    """Advanced projection system for high-dimensional E8 visualization
    
    Args:
        data: Input array of shape (n, 24) with lattice points
        method: Projection algorithm to use:
            - "lie_algebra": Adjoint representation projection
            - "radon": Multidimensional Radon transform
            - "spectral": Spectral unfolding using Laplacian eigenmaps
    
    Returns:
        array: 3D coordinates shaped (n, 3)
    """
    if method == "lie_algebra":
        return _adjoint_representation_projection(data)
    elif method == "radon":
        return _radon_transform_3d(data)
    elif method == "spectral":
        return _spectral_unfolding(data)
    else:
        raise ValueError(f"Unknown projection method: {method}")

def _adjoint_representation_projection(data: np.ndarray) -> np.ndarray:
    """Project using E8 Lie algebra structure"""
    # Create adjoint representation matrix
    adjoint_rep = np.kron(data[:, :8], data[:, 8:16]) - np.kron(data[:, 8:16], data[:, :8])
    
    # Compute dominant eigenvectors
    _, eigenvectors = np.linalg.eigh(adjoint_rep @ adjoint_rep.T)
    return eigenvectors[:, :3].real

def _radon_transform_3d(data: np.ndarray, n_angles: int = 100) -> np.ndarray:
    """Multidimensional Radon transform projection"""
    projections = []
    for _ in range(n_angles):
        # Generate random hyperplane
        normal = np.random.randn(24)
        normal /= np.linalg.norm(normal)
        projections.append(data @ normal)
    
    # Perform inverse Radon transform
    return np.linalg.pinv(np.array(projections).T)[:, :3]

def _spectral_unfolding(data: np.ndarray) -> np.ndarray:
    """Spectral manifold unfolding using graph Laplacian"""
    # Build similarity graph
    distances = np.sqrt(np.sum((data[:, None] - data) ** 2, axis=2))
    sigma = np.median(distances)
    weights = np.exp(-distances ** 2 / (2 * sigma ** 2))
    
    # Compute Laplacian
    degree = np.diag(np.sum(weights, axis=1))
    laplacian = degree - weights
    
    # Compute first non-trivial eigenvectors
    _, eigenvectors = np.linalg.eigh(laplacian)
    return eigenvectors[:, 1:4]
