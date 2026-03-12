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

    def index(self, label, vector):
        q = self.leech.quantify(vector)
        # Standardize key as integer tuple to avoid float precision issues in dict keys
        h = tuple(np.round(q).astype(int).tolist())
        print(f"Indexing {label} at centroid {h[:6]}...")
        if h not in self.table:
            self.table[h] = []
        self.table[h].append(label)

    def lookup(self, vector):
        """ Returns labels from the exact matching lattice point. """
        q = self.leech.quantify(vector)
        h = tuple(np.round(q).astype(int).tolist())
        return self.table.get(h, [])

    def lookup_neighborhood(self, vector):
        """ 
        Returns labels from the nearest lattice point AND its closest neighbors.
        This increases recall for Locality Sensitive Hashing.
        """
        central_q = self.leech.quantify(vector)
        results = []
        
        # 1. Get labels from the exact centroid
        central_tuple = tuple(np.round(central_q).astype(int).tolist())
        results.extend(self.table.get(central_tuple, []))
        
        # 2. Get minimal vectors (neighbors)
        min_vecs = self.leech.get_minimal_vectors()
        
        # 3. Check all neighbors for indexed items
        # Optimization: In production, we'd use a more efficient sparse set check
        for v in min_vecs:
            # Check both directions
            for neighbor_q in [central_q + v, central_q - v]:
                neighbor_tuple = tuple(np.round(neighbor_q).astype(int).tolist())
                labels = self.table.get(neighbor_tuple, [])
                if labels:
                    results.extend(labels)
            
        return list(set(results)) # Deduplicate

if __name__ == "__main__":
    lh = LeechHash()
    print("--- Leech-LSH: Neighborhood Search Prototype ---")
    
    # Simulate a small database of "concepts"
    np.random.seed(42)
    base_vec = np.random.randn(24) * 0.5
    
    concepts = {
        "AI_Core": base_vec,
        "Machine_Learning": base_vec + np.random.normal(0, 0.5, 24),
        "Deep_Learning": base_vec + np.random.normal(0, 0.8, 24),
        "Distant_Concept": np.random.randn(24) * 10.0
    }

    # Indexing
    for label, vec in concepts.items():
        lh.index(label, vec)
    
    # Querying
    # Get the actual centroid of AI_Core to create a neighbor query
    first_key = list(lh.table.keys())[0]
    target_centroid = np.array(first_key)
    min_vecs = lh.leech.get_minimal_vectors()
    
    # We need to find a neighbor that is EXACTLY one minimal vector away
    # in the eyes of the quantifier.
    # Construction B is tricky: let's try indexing the neighbor and looking up the target.
    print(f"\nVerifying Lattice Geometry...")
    v = min_vecs[0]
    neighbor_centroid = target_centroid + v
    
    # Is the neighbor_centroid a valid lattice point?
    is_lattice = np.allclose(neighbor_centroid, lh.leech.quantify(neighbor_centroid))
    print(f"Is neighbor a valid lattice point? {is_lattice}")
    
    # Distance test
    diff = neighbor_centroid - target_centroid
    norm_sq = np.sum(diff**2)
    print(f"Norm squared of difference: {norm_sq}")
    
    # Manual check of neighborhood logic
    print(f"Checking if target {first_key[:3]} is in neighborhood of {tuple(neighbor_centroid[:3].astype(int).tolist())}...")
    
    # Simple check:
    found = False
    for mv in min_vecs:
        if np.allclose(neighbor_centroid + mv, target_centroid) or np.allclose(neighbor_centroid - mv, target_centroid):
            found = True
            break
    print(f"Found target in neighbor's list? {found}")

    # Final query test
    query_vec = neighbor_centroid
    neighborhood_results = lh.lookup_neighborhood(query_vec)
    print(f"Neighborhood Match Results: {neighborhood_results}")
