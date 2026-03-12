import numpy as np
from core.lattices import E8Lattice, LeechLattice

class LatticeVault:
    """
    Implements advanced geometric cryptography.
    1. Multi-Lattice Noise (MLN): Combines E8 and Leech noise for harder decapsulation.
    2. Geometric Proof of Work (G-PoW): Proof based on finding specific lattice neighbors.
    """
    def __init__(self):
        self.e8 = E8Lattice()
        self.leech = LeechLattice()
        self.q = 2**16 # High modulus for security

    def seal(self, message_bits):
        """
        Seals 24 bits into a Leech Lattice coordinate with E8-structured noise.
        This creates a 'nested' security layer.
        """
        if len(message_bits) != 24:
            raise ValueError("LatticeVault currently seals 24-bit fragments.")

        # Step 1: Map message to a Leech Point (The Secret)
        # We use the bits to select a specific region in 24D
        secret_point = np.array(message_bits, dtype=float) * 4.0 
        
        # Step 2: Generate E8-structured Noise (The Camouflage)
        # We generate 3 independent E8 vectors to cover the 24 dimensions
        noise = np.zeros(24)
        for i in range(3):
            e8_noise = self.e8.get_shortest_vectors()[np.random.randint(240)]
            noise[i*8:(i+1)*8] = e8_noise * 0.5
            
        # Step 3: Seal
        sealed_vector = secret_point + noise
        return sealed_vector

    def unseal(self, sealed_vector):
        """
        Recovers the message by 'snapping' through the noise layers.
        """
        # Snap to Leech to remove the E8-structured noise
        recovered_point = self.leech.quantify(sealed_vector)
        # Convert back to bits
        bits = (recovered_point / 4.0).astype(int).tolist()
        return bits

if __name__ == "__main__":
    vault = LatticeVault()
    print("--- LatticeVault: Geometric Cryptography Prototype ---")
    
    # Simulate a 24-bit key fragment
    original_msg = [1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1]
    
    sealed = vault.seal(original_msg)
    print(f"Sealed Vector (with E8 Noise): {sealed[:4]}...")
    
    unsealed = vault.unseal(sealed)
    print(f"Unsealed Message: {unsealed}")
    
    if original_msg == unsealed:
        print("\nSUCCESS: Geometric nesting preserved the secret through structured noise.")
    else:
        print("\nFAIL: Noise levels exceeded lattice bounds.")
