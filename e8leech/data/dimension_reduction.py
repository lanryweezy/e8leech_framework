import numpy as np

class PCADimensionReduction:
    """
    A class for dimension reduction using PCA.
    """

    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        """
        Fits the PCA model to the data.
        """
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Calculate the covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)

        # Calculate the eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort the eigenvectors by eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]

        # Store the principal components
        self.components = eigenvectors[:, :self.n_components]

    def transform(self, X):
        """
        Transforms the data to a lower dimension.
        """
        # Center the data
        X_centered = X - self.mean

        # Project the data onto the principal components
        return np.dot(X_centered, self.components)
