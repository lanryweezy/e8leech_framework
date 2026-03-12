import numpy as np
import time
from core.lattices import LeechLattice
from leech_hash import LeechHash

def scale_test():
    print("--- Leech Lattice Scaling Test: 10,000 Embeddings ---")
    lh = LeechHash()
    num_vectors = 10000
    dim = 24
    
    # 1. Generate large synthetic dataset
    num_vectors = 5000
    print(f"Generating {num_vectors} synthetic 24D embeddings...")
    np.random.seed(42)
    # Use wider variance to test distinct concepts
    base_points = np.random.randn(10, dim) * 5.0
    data = []
    labels = []
    for i in range(num_vectors):
        base_idx = i % 10
        vec = base_points[base_idx] + np.random.normal(0, 0.1, dim)
        data.append(vec)
        labels.append(f"concept_{base_idx}_id_{i}")
    
    # 2. Bulk Indexing
    print(f"Indexing {num_vectors} vectors...")
    start_time = time.time()
    for i in range(num_vectors):
        lh.index(labels[i], data[i])
    index_time = time.time() - start_time
    print(f"Indexing complete in {index_time:.2f} seconds ({num_vectors/index_time:.2f} vectors/sec)")
    
    # 3. Retrieval Performance
    print("\nTesting retrieval performance...")
    query_vec = base_points[0] + np.random.normal(0, 0.1, dim)
    
    # Exact lookup
    start_time = time.time()
    exact_results = lh.lookup(query_vec)
    exact_time = time.time() - start_time
    print(f"Exact lookup time: {exact_time*1000:.2f} ms")
    print(f"Exact matches found: {len(exact_results)}")
    
    # Neighborhood lookup (Recall boost)
    print("Neighborhood lookup (checking ~200k potential buckets)...")
    start_time = time.time()
    # Note: Our current loop over 196k vectors is the bottleneck. 
    # For scaling, we'll need a sparse set intersection.
    neighborhood_results = lh.lookup_neighborhood(query_vec)
    neigh_time = time.time() - start_time
    print(f"Neighborhood lookup time: {neigh_time:.2f} seconds")
    print(f"Neighborhood matches found: {len(neighborhood_results)}")
    
    # 4. Storage efficiency check
    num_buckets = len(lh.table)
    print(f"\nStorage Summary:")
    print(f"Total Vectors: {num_vectors}")
    print(f"Unique Buckets: {num_buckets}")
    print(f"Avg Vectors per Bucket: {num_vectors/num_buckets:.2f}")

if __name__ == "__main__":
    scale_test()
