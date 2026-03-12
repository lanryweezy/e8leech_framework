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
    mapper_e8 = LatticeEmbeddingMapper(lattice_type='e8')
    print("Testing LEM with E8 Lattice...")
    
    # Simulate a batch of 8D embeddings
    np.random.seed(42)
    batch_e8 = np.random.randn(5, 8) * 0.5
    mapped_e8 = mapper_e8.map_embeddings(batch_e8)
    
    distortion_e8 = mapper_e8.calculate_distortion(batch_e8, mapped_e8)
    print(f"E8 Average Distortion (MSE): {distortion_e8:.4f}")
    
    print("\n" + "="*30)
    print("Testing LEM with Leech Lattice (24D)...")
    mapper_leech = LatticeEmbeddingMapper(lattice_type='leech')
    
    # Simulate 24D embeddings (thought vectors)
    # Creating two similar vectors to test relationship preservation
    v1 = np.random.randn(24) * 0.2
    v2 = v1 + np.random.normal(0, 0.05, 24) # v2 is close to v1
    
    batch_leech = np.array([v1, v2])
    mapped_leech = mapper_leech.map_embeddings(batch_leech)
    
    dist_orig = np.linalg.norm(v1 - v2)
    dist_mapped = np.linalg.norm(mapped_leech[0] - mapped_leech[1])
    
    print(f"Original Distance: {dist_orig:.4f}")
    print(f"Mapped Distance:   {dist_mapped:.4f}")
    print(f"Leech Distortion (MSE): {mapper_leech.calculate_distortion(batch_leech, mapped_leech):.4f}")
    
    if dist_mapped == 0 or dist_mapped < dist_orig * 1.5:
        print("SUCCESS: Leech Lattice preserved semantic proximity.")
    else:
        print("INFO: Leech Lattice mapped points to different centroids (high sensitivity).")
