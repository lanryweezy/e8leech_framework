import numpy as np

class LWE:
    """
    Learning with Errors (LWE) based key exchange protocol.
    """
    def __init__(self, n, q, std_dev):
        """
        Initializes the LWE parameters.

        Args:
            n: The dimension of the lattice.
            q: The modulus.
            std_dev: The standard deviation of the error distribution.
        """
        self.n = n
        self.q = q
        self.std_dev = std_dev

    def generate_keys(self):
        """
        Generates a public and private key pair.
        """
        s = np.random.randint(0, self.q, self.n)
        A = np.random.randint(0, self.q, (self.n, self.n))
        e = np.random.normal(0, self.std_dev, self.n).astype(int)
        b = (np.dot(A, s) + e) % self.q
        return (A, b), s

    def key_exchange(self, pk_a, pk_b):
        """
        Performs a key exchange.
        """
        A_a, b_a = pk_a
        A_b, b_b = pk_b

        # Alice computes her shared secret
        s_a = np.random.randint(0, self.q, self.n)
        e_a = np.random.normal(0, self.std_dev, self.n).astype(int)
        c_a = (np.dot(A_b.T, s_a) + e_a) % self.q

        # Bob computes his shared secret
        s_b = np.random.randint(0, self.q, self.n)
        e_b = np.random.normal(0, self.std_dev, self.n).astype(int)
        c_b = (np.dot(A_a.T, s_b) + e_b) % self.q

        # Alice computes the shared key
        k_a = (np.dot(b_b, s_a) - np.dot(c_b, s_a)) % self.q

        # Bob computes the shared key
        k_b = (np.dot(b_a, s_b) - np.dot(c_a, s_b)) % self.q

        return k_a, k_b
