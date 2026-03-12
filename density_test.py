import numpy as np
import sys
import pickle
from core.lattices import LeechLattice
from lem_prototype import LatticeEmbeddingMapper

def test_data_density():
    print("--- Data Density & Compression Analysis ---")
    mapper = LatticeEmbeddingMapper(lattice_type='leech')
    # Scale and reduce sample size for speed
    num_vectors = 100
    dim = 24
    
    # 1. Create raw 24D floating point vectors (standard AI embeddings)
    # Scale raw data to be near the lattice origin for faster quantization
    np.random.seed(42)
    raw_data = (np.random.randn(num_vectors, dim) * 2.0).astype(np.float32)
    raw_size = raw_data.nbytes
    
    # 2. Map to Leech Lattice points
    print(f"Quantizing {num_vectors} vectors to Leech centroids...")
    mapped_data = mapper.map_embeddings(raw_data)
    
    # 3. Calculate compressed representation size
    # In a real system, we'd store indices or small integer offsets.
    # For this test, we convert the unique lattice points to a lookup table
    # and store only the indices.
    unique_points, inverse_indices = np.unique(mapped_data, axis=0, return_inverse=True)
    
    # Size calculation: Lookup Table + Index Array
    # Indices for 1000 points can fit in 16-bit ints
    index_array_size = inverse_indices.astype(np.uint16).nbytes
    lookup_table_size = unique_points.astype(np.int8).nbytes # Lattice points are small integers
    
    total_compressed_size = index_array_size + lookup_table_size
    
    # 4. Results
    print(f"\nResults for {num_vectors} vectors:")
    print(f"Raw FP32 Size:      {raw_size} bytes")
    print(f"Compressed Size:    {total_compressed_size} bytes")
    print(f"Compression Ratio:  {raw_size / total_compressed_size:.2f}x")
    print(f"Space Saved:        {100 - (total_compressed_size/raw_size * 100):.2f}%")
    print(f"Unique Centroids:   {len(unique_points)}")

if __name__ == "__main__":
    test_data_density()
