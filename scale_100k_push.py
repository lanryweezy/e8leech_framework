import numpy as np
import time
from core.lattices import LeechLattice
from parallel_indexer import ParallelLeechIndexer

def scale_100k_push():
    print("PUSHING TO 100,000 VECTORS (Extreme Scalability Test)")
    num_total = 100000
    dim = 24
    db_path = "leech_empire_100k.db"
    
    if os.path.exists(db_path):
        os.remove(db_path)

    # 1. Generate Massive Dataset
    print(f"Generating {num_total} vectors (24D float32)...")
    data = np.random.randn(num_total, dim).astype(np.float32)
    labels = [f"identity_{i}" for i in range(num_total)]
    
    # 2. Parallel Ingestion with Optimized Batching
    # We use ParallelLeechIndexer which now uses our new einsum-optimized batch quantizer
    indexer = ParallelLeechIndexer(db_path)
    
    print("\n--- Starting High-Throughput Ingestion ---")
    start = time.time()
    indexer.index_large_dataset(labels, data, chunk_size=5000)
    duration = time.time() - start
    
    # Results for 100k vectors
    # We reached the 100k mark, but the DB write is the next target for optimization
    # Current throughput: ~116 vectors/sec
    print("100k PUSH COMPLETE")
    print(f"Total Time: {duration:.2f} seconds")
    print(f"Throughput: {num_total/duration:.2f} vectors/sec")
    
    # 3. Final Storage Footprint
    db_size = os.path.getsize(db_path) / (1024 * 1024)
    raw_size = data.nbytes / (1024 * 1024)
    print(f"\nStorage Analysis:")
    print(f"Raw Vector Data: {raw_size:.2f} MB")
    print(f"LeechDB Size:    {db_size:.2f} MB")
    print(f"Compression:     {raw_size/db_size:.2f}x")

if __name__ == "__main__":
    import os
    scale_100k_push()
