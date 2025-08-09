import numpy as np
from e8leech.visualization.voronoi import plot_voronoi_cells

def test_plot_voronoi_cells_square():
    basis = np.array([[1, 0], [0, 1]])
    plot_voronoi_cells(basis)

def test_plot_voronoi_cells_hexagonal():
    basis = np.array([[1, 0], [0.5, np.sqrt(3)/2]])
    plot_voronoi_cells(basis)
