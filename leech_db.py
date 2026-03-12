import sqlite3
import numpy as np
import json
from core.lattices import LeechLattice

class LeechDB:
    """
    Persistent storage for Leech Lattice indexed embeddings using SQLite.
    """
    def __init__(self, db_path="leech_index.db"):
        self.leech = LeechLattice()
        self.conn = sqlite3.connect(db_path)
        self._setup_db()

    def _setup_db(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS buckets (
                centroid_id TEXT PRIMARY KEY,
                labels TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_centroid ON buckets(centroid_id)")
        self.conn.commit()

    def _centroid_to_key(self, centroid):
        return ",".join(map(str, np.round(centroid).astype(int)))

    def index_batch(self, labels, vectors):
        print(f"Quantizing batch of {len(vectors)}...", flush=True)
        centroids = self.leech.quantify_batch(vectors)
        
        print("Writing to database...", flush=True)
        cursor = self.conn.cursor()
        
        # Optimize by grouping labels by bucket to minimize DB operations
        bucket_data = {}
        for label, centroid in zip(labels, centroids):
            key = self._centroid_to_key(centroid)
            if key not in bucket_data:
                bucket_data[key] = []
            bucket_data[key].append(label)
            
        for key, new_labels in bucket_data.items():
            cursor.execute("SELECT labels FROM buckets WHERE centroid_id = ?", (key,))
            row = cursor.fetchone()
            
            if row:
                existing = json.loads(row[0])
                updated = list(set(existing + new_labels))
                cursor.execute("UPDATE buckets SET labels = ? WHERE centroid_id = ?", 
                             (json.dumps(updated), key))
            else:
                cursor.execute("INSERT INTO buckets (centroid_id, labels) VALUES (?, ?)", 
                             (key, json.dumps(new_labels)))
        
        self.conn.commit()

    def query_exact(self, vector):
        centroid = self.leech.quantify(vector)
        key = self._centroid_to_key(centroid)
        cursor = self.conn.cursor()
        cursor.execute("SELECT labels FROM buckets WHERE centroid_id = ?", (key,))
        row = cursor.fetchone()
        return json.loads(row[0]) if row else []

    def query_neighborhood(self, vector):
        """ 
        Finds all labels in the nearest lattice point and all its neighbors.
        Highly optimized using vectorized distance check against occupied buckets.
        """
        central_q = self.leech.quantify(vector)
        
        # 1. Get all occupied bucket keys from the DB
        cursor = self.conn.cursor()
        cursor.execute("SELECT centroid_id FROM buckets")
        rows = cursor.fetchall()
        if not rows:
            return []
            
        # Convert string keys back to numpy arrays
        def key_to_arr(k):
            return np.array([int(x) for x in k.split(",")])
            
        occupied_keys_arr = np.array([key_to_arr(row[0]) for row in rows])
        
        # 2. Vectorized distance check
        diffs = occupied_keys_arr - central_q
        dists_sq = np.sum(diffs**2, axis=1)
        
        # Leech minimal vectors have norm^2 = 32
        # We also include distance 0 (the central bucket itself)
        matching_indices = np.where((dists_sq < 33.0) & ((np.abs(dists_sq - 32.0) < 1e-5) | (dists_sq < 1e-5)))[0]
        
        results = []
        for idx in matching_indices:
            key = ",".join(map(str, occupied_keys_arr[idx]))
            cursor.execute("SELECT labels FROM buckets WHERE centroid_id = ?", (key,))
            results.extend(json.loads(cursor.fetchone()[0]))
                    
        return list(set(results))

    def close(self):
        self.conn.close()

if __name__ == "__main__":
    db = LeechDB("test_scale.db")
    print("--- LeechDB: Persistent Scaling Test ---")
    
    # Generate 1000 vectors
    num = 1000
    data = np.random.randn(num, 24) * 5.0
    labels = [f"item_{i}" for i in range(num)]
    
    db.index_batch(labels, data)
    
    # Query one
    results = db.query_exact(data[0])
    print(f"Query for item_0 results: {results[:5]}... (Total: {len(results)})")
    db.close()
