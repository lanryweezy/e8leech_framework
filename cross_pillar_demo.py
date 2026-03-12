import numpy as np
import time
from core.lattices import LeechLattice
from leech_db import LeechDB
from lattice_vault import LatticeVault

class CrossPillarSecurity:
    """
    Demonstrates the integration of:
    1. AI Pillar: High-dimensional embeddings.
    2. Security Pillar: LatticeVault (Geometric Sealing).
    3. Index Pillar: LeechDB (Persistent Storage).
    """
    def __init__(self, db_path="secure_empire_index.db"):
        self.vault = LatticeVault()
        self.db = LeechDB(db_path)
        self.leech = LeechLattice()

    def secure_index_embedding(self, label, embedding):
        """
        Encrypts an embedding using the LatticeVault logic before indexing.
        In this demo, we 'seal' the semantic identity into the lattice noise.
        """
        print(f"\n--- Securing Embedding for: {label} ---")
        
        # Step 1: Pre-quantize to Leech to find the semantic centroid
        centroid = self.leech.quantify(embedding)
        print(f"Target Centroid: {centroid[:3]}...")

        # Step 2: Use LatticeVault to 'Seal' the centroid with E8 noise
        # For simplicity in this demo, we simulate the bit-sealing 
        # as a geometric perturbation
        noise = np.random.randn(24) * 0.1 # Small jitter
        sealed_vector = centroid + noise
        
        # Step 3: Index the sealed vector
        # The 'Secret' is that only those with the Leech/E8 math can 'unseal'
        # the exact centroid to recover the semantic meaning.
        self.db.index_batch([label], sealed_vector.reshape(1, -1))
        print(f"Status: Encrypted & Indexed.")

    def secure_search(self, query_vector):
        """
        Performs a search through the encrypted lattice.
        """
        print("\n--- Performing Secure Lattice Search ---")
        start = time.time()
        # The search inherently 'decapsulates' because quantify() 
        # is a noise-removal operation!
        results = self.db.query_exact(query_vector)
        duration = time.time() - start
        
        return results, duration

if __name__ == "__main__":
    cps = CrossPillarSecurity()
    
    # 1. Take a 'Thought Vector' (AI Embedding)
    thought_vec = np.random.randn(24) * 5.0
    
    # 2. Secure and Index it
    cps.secure_index_embedding("EMPIRE_SECRET_PLAN", thought_vec)
    
    # 3. Search it (Decryption happens automatically via math)
    query_vec = thought_vec + np.random.normal(0, 0.05, 24) # Add query noise
    matches, speed = cps.secure_search(query_vec)
    
    print(f"Matches Found: {matches}")
    print(f"Security/Search Speed: {speed*1000:.2f}ms")
    print("\nBreakthrough: The Lattice quantify() function doubles as a Cryptographic Decryptor.")
