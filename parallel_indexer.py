import numpy as np
import time
import multiprocessing as mp
from core.lattices import LeechLattice
from leech_db import LeechDB
import os

def _worker_quantize(chunk):
    """Worker function to quantize a chunk of vectors."""
    # Each worker creates its own lattice instance to avoid sharing state if any
    leech = LeechLattice()
    return leech.quantify_batch(chunk)

class ParallelLeechIndexer:
    """
    High-performance indexer using multiprocessing to saturate CPU cores
    for the 4096-coset Leech math.
    """
    def __init__(self, db_path="leech_parallel.db", num_workers=None):
        self.db_path = db_path
        self.num_workers = num_workers or mp.cpu_count()
        print(f"Parallel Indexer initialized with {self.num_workers} workers.")

    def index_large_dataset(self, labels, vectors, chunk_size=1000):
        """
        Splits a massive dataset into chunks and processes them in parallel.
        """
        db = LeechDB(self.db_path)
        num_total = len(vectors)
        print(f"Starting parallel index of {num_total} vectors...")
        
        start_time = time.time()
        
        # Split data into chunks for workers
        # We use a larger chunk size for workers to minimize IPC overhead
        worker_chunk_size = max(100, num_total // (self.num_workers * 2))
        chunks = [vectors[i:i + worker_chunk_size] for i in range(0, num_total, worker_chunk_size)]
        
        print(f"Processing {len(chunks)} worker chunks...")
        
        with mp.Pool(processes=self.num_workers) as pool:
            # Step 1: Quantize in parallel
            all_centroids = pool.map(_worker_quantize, chunks)
            
        # Flatten results
        centroids = np.vstack(all_centroids)
        
        # Step 2: Sequential DB write (using precomputed centroids)
        print("Quantization complete. Writing to LeechDB...")
        db_batch_size = chunk_size
        for i in range(0, num_total, db_batch_size):
            end_idx = min(i + db_batch_size, num_total)
            db.index_batch_precomputed(labels[i:end_idx], centroids[i:end_idx])
            
        total_duration = time.time() - start_time
        print(f"Parallel Indexing Complete: {total_duration:.2f}s ({num_total/total_duration:.2f} vectors/sec)")
        db.close()

if __name__ == "__main__":
    # Test with 20,000 vectors to verify speedup
    num_test = 20000
    dim = 24
    test_data = np.random.randn(num_test, dim).astype(np.float32)
    test_labels = [f"par_item_{i}" for i in range(num_test)]
    
    indexer = ParallelLeechIndexer("leech_scale_test.db")
    indexer.index_large_dataset(test_labels, test_data)
