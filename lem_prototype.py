import numpy as np
from core.lattices import E8Lattice, LeechLattice

class LatticeEmbeddingMapper:
    """
    Prototypes the 'Lattice Embedding Mapping' (LEM) system.
    Maps high-dimensional continuous embeddings to discrete lattice points.
    """
    def __init__(self, lattice_type='e8'):
        if lattice_type == 'e8':
            self.lattice = E8Lattice()
        elif lattice_type == 'leech':
            self.lattice = LeechLattice()
        else:
            raise ValueError("Unsupported lattice type. Use 'e8' or 'leech'.")
            
    def map_embeddings(self, embeddings):
        """
        Maps a batch of embeddings to the nearest lattice points.
        embeddings: ndarray of shape (N, dim)
        """
        mapped = []
        for vec in embeddings:
            # Scale or normalize if needed to fit the lattice density
            # For now, we assume the input is already appropriately scaled
            q = self.lattice.quantify(vec)
            mapped.append(q)
        return np.array(mapped)

    def calculate_distortion(self, original, mapped):
        """ Calculates the Mean Squared Error between original and mapped points. """
        return np.mean(np.sum((original - mapped)**2, axis=1))

if __name__ == "__main__":
    # Test with E8
    mapper = LatticeEmbeddingMapper(lattice_type='e8')
    print("Testing LEM with E8 Lattice...")
    
    # Simulate a batch of 8D embeddings
    np.random.seed(42)
    batch = np.random.randn(5, 8) * 0.5
    mapped = mapper.map_embeddings(batch)
    
    distortion = mapper.calculate_distortion(batch, mapped)
    print(f"Average Distortion (MSE): {distortion:.4f}")
    print(f"Mapped Example 0: {mapped[0]}")
