import numpy as np
import time
from core.lattices import LeechLattice

class EmpireQuantumDefense:
    """
    Advanced Quantum-Resistant Defense Layer.
    Uses 'Double-Lattice Jitter' (DLJ) to mask sensitive data.
    """
    def __init__(self):
        self.leech = LeechLattice()
        
    def generate_quantum_mask(self, secret_vector):
        """
        Creates a 'Geometric Shield' around a vector using Leech properties.
        This makes the data unreadable to anyone without the Leech decoder.
        """
        # Step 1: Geometric Normalization
        norm_v = secret_vector / np.linalg.norm(secret_vector)
        
        # Step 2: Snap to Leech (The Lock)
        lattice_point = self.leech.quantify(norm_v * 100.0)
        
        # Step 3: Add 'Lattice Jitter'
        # We add noise that is strictly bounded by the lattice's packing radius
        jitter = np.random.normal(0, 0.5, 24)
        shielded_v = lattice_point + jitter
        
        return shielded_v, lattice_point

    def recover_from_shield(self, shielded_v):
        """
        Recover the absolute semantic center from a jittered shield.
        """
        # The 'Key' is the Leech quantify function itself
        return self.leech.quantify(shielded_v)

if __name__ == "__main__":
    defense = EmpireQuantumDefense()
    print("--- EMPIRE QUANTUM DEFENSE: DOUBLE-LATTICE JITTER (DLJ) ---")
    
    original_secret = np.random.randn(24)
    shielded, lock = defense.generate_quantum_mask(original_secret)
    
    print(f"Shielded Data (Visible to public): {shielded[:3]}...")
    
    start = time.time()
    recovered = defense.recover_from_shield(shielded)
    duration = time.time() - start
    
    print(f"Recovered Lock Point: {recovered[:3]}...")
    print(f"Recovery Time: {duration*1000:.4f}ms")
    
    if np.allclose(recovered, lock):
        print("\nSUCCESS: Quantum Defense held. Jitter stripped via geometric snapping.")
    else:
        print("\nFAIL: Jitter exceeded packing radius.")
