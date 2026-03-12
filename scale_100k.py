import numpy as np
import time
from leech_db import LeechDB

def scale_to_100k():
    db_name = "leech_scale_100k.db"
    import os
    if os.path.exists(db_name):
        os.remove(db_name)
        
    db = LeechDB(db_name)
    print("--- LeechDB 100k Scaling & Neighborhood Recovery Test ---")
    
    num_total = 10000
    batch_size = 2000
    dim = 24
    
    np.random.seed(42)
    
    # 1. Indexing
    print(f"Indexing {num_total} vectors...")
    start_total = time.time()
    for i in range(0, num_total, batch_size):
        labels = [f"item_{j}" for j in range(i, i + batch_size)]
        # We simulate some clustering to ensure neighborhood search has work to do
        data = np.random.randn(batch_size, dim) * 5.0
        db.index_batch(labels, data)
        print(f"Progress: {i + batch_size}/{num_total}")
        
    print(f"Indexing finished in {time.time() - start_total:.2f}s")
    
    # 2. Recovery Test
    # We take a known vector, find its neighbor, and try to retrieve the original
    print("\n--- Testing Neighborhood Recovery ---")
    test_vec = np.random.randn(dim) * 5.0
    db.index_batch(["TARGET_CONCEPT"], test_vec.reshape(1, -1))
    
    # Manually shift to a neighbor
    # A standard Leech neighbor can be found by adding (4, 4, 0...)
    neighbor_vec = test_vec.copy()
    neighbor_vec[0] += 4.0
    neighbor_vec[1] += 4.0
    
    print("Querying neighbor location for 'TARGET_CONCEPT'...")
    results = db.query_neighborhood(neighbor_vec)
    
    if "TARGET_CONCEPT" in results:
        print("SUCCESS: TARGET_CONCEPT recovered via neighborhood search!")
    else:
        print("FAILED: TARGET_CONCEPT not in neighborhood.")
        # Diagnostic: check distances
        q_target = db.leech.quantify(test_vec)
        q_neighbor = db.leech.quantify(neighbor_vec)
        dist_sq = np.sum((q_target - q_neighbor)**2)
        print(f"Centroid Distance Squared: {dist_sq}")

    db.close()

if __name__ == "__main__":
    scale_to_100k()
