import numpy as np
import plotly.graph_objects as go
from e8leech.core.golay_code import LeechLattice

class HologramVisualizer:
    """
    A class for visualizing 24D holograms of the Leech lattice.
    """

    def __init__(self, n_points=100):
        self.leech = LeechLattice()
        self.points = self._generate_leech_points(n_points)

    def _generate_leech_points(self, n_points):
        """
        Generates a sample of points from the Leech lattice.
        This is a simplified approach. A real implementation would be more sophisticated.
        """
        # For simplicity, we'll generate random integer combinations of the basis vectors.
        basis = self.leech.basis
        points = []
        for _ in range(n_points):
            coeffs = np.random.randint(-2, 3, size=24)
            points.append(np.dot(coeffs, basis))
        return np.array(points)

    def project_to_3d(self, points):
        """
        Projects the 24D points to 3D using PCA.
        """
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        return pca.fit_transform(points)

    def display_interactive_hologram(self):
        """
        Displays an interactive 3D hologram of the Leech lattice points.
        """
        points_3d = self.project_to_3d(self.points)

        fig = go.Figure(data=[go.Scatter3d(
            x=points_3d[:, 0],
            y=points_3d[:, 1],
            z=points_3d[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=points_3d[:, 2], # Color by z-coordinate
                colorscale='Viridis',
                opacity=0.8
            )
        )])

        fig.update_layout(
            title='Interactive Hologram of the Leech Lattice',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )
        fig.show()
