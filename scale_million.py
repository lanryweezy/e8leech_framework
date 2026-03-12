import numpy as np
import time
import os
from leech_db import LeechDB
from parallel_indexer import ParallelLeechIndexer

def push_million_vectors():
    print("--- EMPIRE SCALE: 1,000,000 VECTOR INGESTION TEST ---")
    db_path = "leech_empire_million.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    db = LeechDB(db_path)
    num_total = 1000000
    batch_size = 50000 # Large batches to leverage the SQL Staging Table optimization
    dim = 24
    
    np.random.seed(42)
    
    print(f"Target: {num_total} vectors across {num_total // batch_size} major bulk commits.")
    start_total = time.time()
    
    for i in range(0, num_total, batch_size):
        chunk_labels = [f"identity_{j}" for j in range(i, i + batch_size)]
        # Simulate structured data clusters to test SQLite group_by performance
        # Using 1000 core "concept" centers
        centers = np.random.randn(1000, dim) * 5.0
        chunk_data = centers[np.random.randint(0, 1000, batch_size)] + np.random.normal(0, 0.1, (batch_size, dim))
        
        start_batch = time.time()
        # Use our new C-optimized staging table engine
        db.index_million_bulk(chunk_labels, chunk_data)
        batch_time = time.time() - start_batch
        print(f"Bulk Commit {i//batch_size + 1}: {batch_size} vectors in {batch_time:.2f}s")
        
    total_time = time.time() - start_total
    print(f"\nSUCCESS: 1,000,000 Vector Ingestion Complete!")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Throughput: {num_total/total_time:.2f} vectors/sec")
    
    db_size = os.path.getsize(db_path) / (1024 * 1024)
    print(f"Final DB Size: {db_size:.2f} MB")
    db.close()

if __name__ == "__main__":
    push_million_vectors()
