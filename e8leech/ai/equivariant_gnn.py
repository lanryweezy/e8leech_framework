import torch
import torch.nn as nn
from e8leech.core.golay_code import E8Lattice, LeechLattice

class EquivariantGNN(nn.Module):
    """
    An Equivariant Graph Neural Network for the E8 and Leech lattices.
    """

    def __init__(self, in_features, out_features, lattice_type='E8'):
        super(EquivariantGNN, self).__init__()
        if lattice_type == 'E8':
            self.lattice = E8Lattice()
        elif lattice_type == 'Leech':
            self.lattice = LeechLattice()
        else:
            raise ValueError("Unsupported lattice type")

        self.linear = nn.Linear(in_features, out_features)

    def forward(self, data):
        """
        Forward pass of the Equivariant GNN.
        data: A batch of graphs from PyTorch Geometric.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # A simple equivariant layer: W * x
        # For full equivariance, the weights W must commute with the symmetry group action.
        # This is a simplified implementation.

        # Apply linear transformation
        x = self.linear(x)

        # Aggregate features from neighbors
        row, col = edge_index
        x_aggregated = torch.zeros_like(x)
        x_aggregated.index_add_(0, row, x[col])

        return x_aggregated

    def to_torchscript(self):
        """
        Exports the model to TorchScript.
        """
        return torch.jit.script(self)

    def to_onnx(self, dummy_input, filepath):
        """
        Exports the model to ONNX format.
        """
        torch.onnx.export(self, dummy_input, filepath, export_params=True, opset_version=10, do_constant_folding=True)
