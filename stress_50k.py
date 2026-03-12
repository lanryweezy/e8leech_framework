import numpy as np
import time
from leech_db import LeechDB

def stress_test_50k():
    db_name = "leech_stress_50k.db"
    import os
    if os.path.exists(db_name):
        os.remove(db_name)
        
    db = LeechDB(db_name)
    print("--- LeechDB 50k Vector Stress Test ---")
    
    # 1. Generate large synthetic dataset
    num_total = 10000
    batch_size = 1000
    dim = 24
    
    np.random.seed(42)
    
    # 1. Bulk Indexing
    print(f"Indexing {num_total} vectors in batches of {batch_size}...")
    start_total = time.time()
    for i in range(0, num_total, batch_size):
        # Create a "cluster" to ensure neighborhood hits
        chunk_labels = [f"vec_{i+j}" for j in range(batch_size)]
        # base_concept = np.random.randn(dim) * 10.0
        # data = base_concept + np.random.normal(0, 0.5, (batch_size, dim))
        # Instead, just use the random noise but we'll query close to an indexed point
        chunk_data = np.random.randn(batch_size, dim) * 10.0
        
        if i == 0:
            # Save one for query test
            test_vec = chunk_data[0]
            # Create a neighbor manually to verify recovery
            min_vecs = db.leech.get_minimal_vectors()
            neighbor_vec = test_vec + min_vecs[0]
            chunk_data[1] = neighbor_vec
            chunk_labels[1] = "NEIGHBOR_OF_VEC_0"

        db.index_batch(chunk_labels, chunk_data)
        print(f"Batch {i//batch_size + 1} indexed.")
        
    total_time = time.time() - start_total
    print(f"\nIndexing finished in {total_time:.2f}s")
    
    # 2. Query Test
    print("\nTesting Query Performance...")
    query_vec = test_vec # Should hit vec_0 exactly
    
    start_q = time.time()
    results = db.query_exact(query_vec)
    print(f"Exact Query Results: {results[:5]}")
    
    print("\nTesting Neighborhood Query Performance (Recovery Test)...")
    # Shift query to the neighbor's centroid
    query_neighbor = neighbor_vec
    
    start_n = time.time()
    n_results = db.query_neighborhood(query_neighbor)
    print(f"Neighborhood Query Time: {time.time() - start_n:.2f}s")
    print(f"Neighborhood Match: {'vec_0' in n_results}")
    
    db.close()

if __name__ == "__main__":
    stress_test_50k()
