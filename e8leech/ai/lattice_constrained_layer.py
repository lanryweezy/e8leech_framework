import torch
import torch.nn as nn
from e8leech.core.golay_code import E8Lattice, LeechLattice

class LatticeConstrainedLayer(nn.Module):
    """
    A layer that constrains its output to a specific lattice.
    """

    def __init__(self, lattice_type='E8'):
        super(LatticeConstrainedLayer, self).__init__()
        if lattice_type == 'E8':
            self.lattice = E8Lattice()
        elif lattice_type == 'Leech':
            self.lattice = LeechLattice()
        else:
            raise ValueError("Unsupported lattice type")

    def forward(self, x):
        """
        Forward pass of the LatticeConstrainedLayer.
        x: Input tensor
        """
        # Project the input tensor onto the lattice
        # This is done by finding the closest lattice point to the input.

        # The 'closest_vector' method needs to be implemented for LeechLattice as well.
        # For now, we assume it exists.

        # The input x is a batch of vectors. We need to process each one.
        constrained_x = torch.zeros_like(x)
        for i in range(x.shape[0]):
            constrained_x[i] = torch.from_numpy(self.lattice.closest_vector(x[i].detach().numpy()))

        return constrained_x
