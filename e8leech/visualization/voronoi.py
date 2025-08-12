import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

def plot_voronoi_cells(basis):
    """
    Plots the Voronoi cells of a 2D lattice.
    """
    if basis.shape[1] != 2:
        raise ValueError("The basis must be 2-dimensional")

    points = np.array([i * basis[0] + j * basis[1] for i in range(-5, 6) for j in range(-5, 6)])

    vor = Voronoi(points)

    fig = voronoi_plot_2d(vor)
    plt.show()
