import numpy as np

class LatticeKeyExchange:
    """
    A key exchange protocol based on the hardness of the Shortest Vector Problem (SVP).
    """
    def __init__(self, lattice_basis):
        """
        Initializes the key exchange protocol.

        Args:
            lattice_basis: The basis of the lattice to use.
        """
        self.lattice_basis = lattice_basis
        self.n = lattice_basis.shape[1]

    def generate_private_key(self):
        """
        Generates a private key.
        """
        # The private key is a short vector in the lattice.
        # We can generate a short vector by taking a small integer linear combination
        # of the basis vectors.
        c = np.random.randint(-10, 10, self.n)
        return np.dot(c, self.lattice_basis)

    def generate_public_key(self, private_key):
        """
        Generates a public key from a private key.
        """
        # The public key is the private key with some noise added.
        # The noise is a small vector that is not in the lattice.
        noise = np.random.normal(0, 0.1, self.n)
        return private_key + noise

    def get_shared_secret(self, private_key, public_key):
        """
        Computes the shared secret.
        """
        # The shared secret is computed by taking the dot product of the private key
        # with the public key, and then rounding to the nearest integer.
        # This is a simplified version of a real-world key exchange protocol.
        return int(np.round(np.dot(private_key, public_key)))
