import numpy as np
from e8leech.crypto.lwe import LWE

class LWEKeyExchange:
    """
    A class for a key exchange protocol based on LWE.
    """

    def __init__(self, n, q, std_dev=1.0):
        self.lwe = LWE(n, q, std_dev)

    def get_public_key(self):
        """
        Returns the public key.
        """
        return self.lwe.public_key

    def get_shared_secret(self, other_public_key):
        """
        Generates a shared secret given the other party's public key.
        """
        # This is a simplified key exchange protocol.
        # A real implementation would be more complex and secure.
        A, b = other_public_key
        s = self.lwe.secret_key

        # Shared secret is approximately b - As
        shared_secret = (b - A @ s) % self.lwe.q

        return shared_secret
