import numpy as np
from core.lattices import LeechLattice
import time

class LeechHash:
    """
    Experimental Lattice-Based Hashing.
    Uses the Leech Lattice to provide locality-sensitive hashing (LSH)
    where similar vectors naturally collide into the same lattice point.
    """
    def __init__(self):
        self.leech = LeechLattice()
        self.table = {}

    def hash_vector(self, vector):
        """ Returns a string representation of the nearest Leech Lattice point. """
        q = self.leech.quantify(vector)
        # Convert the lattice point to a tuple key for hashing
        return tuple(q.tolist())

    def index(self, label, vector):
        h = self.hash_vector(vector)
        if h not in self.table:
            self.table[h] = []
        self.table[h].append(label)

    def lookup(self, vector):
        h = self.hash_vector(vector)
        return self.table.get(h, [])

if __name__ == "__main__":
    lh = LeechHash()
    print("--- Leech-LSH: Locality Sensitive Hashing Prototype ---")
    
    # Simulate a small database of "concepts"
    concepts = {
        "deep_learning": np.random.randn(24) * 0.1,
        "neural_networks": None, # Will be set to deep_learning + noise
        "quantum_computing": np.random.randn(24) * 5.0, # Far away
        "cryptography": np.random.randn(24) * 5.2
    }
    concepts["neural_networks"] = concepts["deep_learning"] + np.random.normal(0, 0.02, 24)

    # Indexing
    for label, vec in concepts.items():
        lh.index(label, vec)
    
    # Querying
    print("\nQuerying with small noise (0.001)...")
    query_vec = concepts["deep_learning"] + np.random.normal(0, 0.001, 24)
    results = lh.lookup(query_vec)
    
    print(f"Query Results: {results}")
    
    # Let's see the unique hash keys
    print(f"\nTotal Hash Buckets: {len(lh.table)}")
    for h, labels in lh.table.items():
        print(f"Bucket {h[:3]}... : {labels}")
