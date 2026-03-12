import numpy as np
from core.lattices import LeechLattice
from leech_db import LeechDB

class SemanticRouter:
    """
    Routes high-dimensional embeddings to specialized 'Expert' handlers
    based on their position in the Leech Lattice.
    """
    def __init__(self, db_path="leech_empire_100k.db"):
        self.leech = LeechLattice()
        self.db = LeechDB(db_path)
        self.experts = {} # Map of centroid_id -> expert_label

    def register_expert(self, expert_label, example_vectors):
        """
        Learns which lattice regions belong to which expert based on examples.
        """
        print(f"Registering expert: {expert_label}...")
        centroids = self.leech.quantify_batch(np.array(example_vectors))
        for c in centroids:
            key = self.db._centroid_to_key(c)
            self.experts[key] = expert_label

    def route(self, vector):
        """
        Snaps the input to the lattice and routes to the nearest registered expert.
        """
        q = self.leech.quantify(vector)
        key = self.db._centroid_to_key(q)
        
        # 1. Direct Hit
        if key in self.experts:
            return self.experts[key], "DIRECT"
        
        # 2. Neighborhood Search (Fuzzy Routing)
        # If the exact point isn't an expert, check neighbors
        min_vecs = self.leech.get_minimal_vectors()
        for v in min_vecs:
            for neighbor in [q + v, q - v]:
                n_key = self.db._centroid_to_key(neighbor)
                if n_key in self.experts:
                    return self.experts[n_key], "NEIGHBORHOOD"
        
        return "GENERAL_MODEL", "FALLBACK"

if __name__ == "__main__":
    router = SemanticRouter()
    
    # Define Expert Regions
    print("--- Training Semantic Router ---")
    router.register_expert("FINANCE_EXPERT", [np.random.randn(24) + 10.0 for _ in range(5)])
    router.register_expert("LEGAL_EXPERT", [np.random.randn(24) - 10.0 for _ in range(5)])
    
    # Test Routing
    test_query = np.random.randn(24) + 10.2 # Near Finance
    expert, mode = router.route(test_query)
    print(f"\nQuery routed to: {expert} (Mode: {mode})")
    
    test_query_fuzzy = np.random.randn(24) + 8.5 # Between General and Finance
    expert_fuzzy, mode_fuzzy = router.route(test_query_fuzzy)
    print(f"Fuzzy query routed to: {expert_fuzzy} (Mode: {mode_fuzzy})")
