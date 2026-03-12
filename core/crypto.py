import numpy as np
from core.lattices import E8Lattice

class LatticeExchange:
    """
    A conceptual implementation of Lattice-based Key Exchange.
    Inspired by Learning With Errors (LWE) and Kyber/ML-KEM principles,
    using the E8 Lattice as a structured noise distribution.
    """
    def __init__(self):
        self.lattice = E8Lattice()
        self.dim = 8
        self.q = 256 # Modulus for the secret integer space

    def generate_keypair(self):
        """ Generates a public/private key pair. """
        # Secret s: small random vector
        secret = np.random.randint(-2, 3, size=self.dim)
        
        # Matrix A: Public random matrix
        A = np.random.randint(0, self.q, size=(self.dim, self.dim))
        
        # Error e: Error sampled from the E8 Lattice
        # We take a small random vector and 'snap' it to E8 to get structured noise
        e = self.lattice.quantify(np.random.randn(self.dim) * 0.5)
        
        # Public key: t = As + e (mod q)
        t = (np.dot(A, secret) + e) % self.q
        
        return secret, (A, t)

    def exchange_info(self):
        return "This module demonstrates structured noise via E8 for PQC-like primitives."

if __name__ == "__main__":
    kex = LatticeExchange()
    s, (A, t) = kex.generate_keypair()
    print("Lattice Key Exchange Prototype (E8-LWE)")
    print(f"Secret Key (s): {s}")
    print(f"Public Key (t): {t[:4]}... (truncated)")
    print("SUCCESS: Public key generated using E8-structured error.")
