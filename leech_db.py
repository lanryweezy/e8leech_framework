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
        print(f"Quantizing batch of {len(vectors)}...")
        centroids = self.leech.quantify_batch(vectors)
        
        cursor = self.conn.cursor()
        for label, centroid in zip(labels, centroids):
            key = self._centroid_to_key(centroid)
            
            # Get existing labels
            cursor.execute("SELECT labels FROM buckets WHERE centroid_id = ?", (key,))
            row = cursor.fetchone()
            
            if row:
                existing = json.loads(row[0])
                if label not in existing:
                    existing.append(label)
                    cursor.execute("UPDATE buckets SET labels = ? WHERE centroid_id = ?", 
                                 (json.dumps(existing), key))
            else:
                cursor.execute("INSERT INTO buckets (centroid_id, labels) VALUES (?, ?)", 
                             (key, json.dumps([label])))
        
        self.conn.commit()

    def query_exact(self, vector):
        centroid = self.leech.quantify(vector)
        key = self._centroid_to_key(centroid)
        cursor = self.conn.cursor()
        cursor.execute("SELECT labels FROM buckets WHERE centroid_id = ?", (key,))
        row = cursor.fetchone()
        return json.loads(row[0]) if row else []

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
