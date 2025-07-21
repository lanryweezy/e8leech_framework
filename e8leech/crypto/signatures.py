import numpy as np
import hashlib

class Bliss:
    """
    BLISS signature scheme.
    """
    def __init__(self, n, q, d, std_dev):
        """
        Initializes the BLISS parameters.

        Args:
            n: The dimension of the lattice.
            q: The modulus.
            d: The number of non-zero elements in the secret key.
            std_dev: The standard deviation of the error distribution.
        """
        self.n = n
        self.q = q
        self.d = d
        self.std_dev = std_dev

    def generate_keys(self):
        """
        Generates a public and private key pair.
        """
        # Generate the secret key s with d non-zero elements.
        s = np.zeros(self.n)
        indices = np.random.choice(self.n, self.d, replace=False)
        s[indices] = np.random.randint(1, self.q, self.d)

        # Generate the public key A.
        A = np.random.randint(0, self.q, (self.n, self.n))
        e = np.random.normal(0, self.std_dev, self.n).astype(int)
        b = (np.dot(A, s) + 2 * e) % self.q

        return (A, b), s

    def sign(self, message, sk):
        """
        Signs a message.
        """
        A, b = sk[0]
        s = sk[1]

        # Hash the message.
        h = hashlib.sha256(message.encode()).hexdigest()
        h_int = int(h, 16)

        # Generate a random vector y.
        y = np.random.randint(0, self.q, self.n)

        # Compute Ay.
        Ay = np.dot(A, y) % self.q

        # Compute the challenge c.
        c = (h_int + Ay) % self.q

        # Compute z = y + (-1)^b * s * c
        b_sign = np.random.randint(0, 2)
        z = (y + ((-1)**b_sign) * s * c) % self.q

        return (z, c, b_sign)

    def verify(self, message, signature, pk):
        """
        Verifies a signature.
        """
        A, b = pk
        z, c, b_sign = signature

        # Hash the message.
        h = hashlib.sha256(message.encode()).hexdigest()
        h_int = int(h, 16)

        # Compute Ay.
        Ay = (np.dot(A, z) - ((-1)**b_sign) * b * c) % self.q

        # Compute the challenge c'.
        c_prime = (h_int + Ay) % self.q

        return c == c_prime
