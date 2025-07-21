import numpy as np

class LWE:
    """
    A class for Learning with Errors (LWE) cryptosystem.
    """

    def __init__(self, n, q, std_dev=1.0):
        """
        Initializes the LWE cryptosystem.
        :param n: The dimension of the lattice.
        :param q: The modulus.
        :param std_dev: The standard deviation of the error distribution.
        """
        self.n = n
        self.q = q
        self.std_dev = std_dev
        self.secret_key = self._generate_secret_key()
        self.public_key = self._generate_public_key()

    def _generate_secret_key(self):
        """
        Generates a secret key.
        """
        return np.random.randint(0, self.q, size=self.n)

    def _generate_public_key(self):
        """
        Generates a public key.
        """
        A = np.random.randint(0, self.q, size=(self.n, self.n))
        e = np.random.normal(0, self.std_dev, size=self.n).astype(int)
        b = (A @ self.secret_key + e) % self.q
        return A, b

    def encrypt(self, message):
        """
        Encrypts a single bit message.
        """
        if message not in [0, 1]:
            raise ValueError("Message must be 0 or 1.")

        A, b = self.public_key
        s = np.random.randint(0, 2, size=self.n)
        u = A.T @ s % self.q
        v = (b.T @ s + message * (self.q // 2)) % self.q
        return u, v

    def decrypt(self, ciphertext):
        """
        Decrypts a ciphertext.
        """
        u, v = ciphertext
        decrypted_val = (v - self.secret_key @ u) % self.q
        return 0 if decrypted_val < self.q / 2 else 1
