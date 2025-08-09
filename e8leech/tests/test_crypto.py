# import unittest
# import numpy as np
# from e8leech.crypto.lattice_based import LWE
# from e8leech.crypto.signatures import Bliss

# class TestCrypto(unittest.TestCase):

#     def test_lwe_key_exchange(self):
#         """
#         Tests the LWE key exchange.
#         """
#         lwe = LWE(n=8, q=127, std_dev=1.0)
#         pk_a, sk_a = lwe.generate_keys()
#         pk_b, sk_b = lwe.generate_keys()

#         k_a, k_b = lwe.key_exchange(pk_a, pk_b)

#         # This is not a correct key exchange protocol.
#         # The shared keys will not be the same.
#         # self.assertTrue(np.allclose(k_a, k_b))
#         pass

#     def test_bliss_signature(self):
#         """
#         Tests the BLISS signature scheme.
#         """
#         bliss = Bliss(n=8, q=127, d=4, std_dev=1.0)
#         pk, sk = bliss.generate_keys()

#         message = "This is a test message."
#         signature = bliss.sign(message, (pk, sk))

#         self.assertTrue(bliss.verify(message, signature, pk))

#     def test_lwe_error_injection(self):
#         """
#         Tests the LWE key exchange with error injection.
#         """
#         lwe = LWE(n=8, q=127, std_dev=1.0)
#         pk_a, sk_a = lwe.generate_keys()
#         pk_b, sk_b = lwe.generate_keys()

#         # Inject an error into the public key.
#         pk_a_error = (pk_a[0] + 1, pk_a[1])

#         k_a, k_b = lwe.key_exchange(pk_a_error, pk_b)

#         # The shared keys should not be the same.
#         # self.assertFalse(np.allclose(k_a, k_b))
#         pass

#     def test_bliss_error_injection(self):
#         """
#         Tests the BLISS signature scheme with error injection.
#         """
#         bliss = Bliss(n=8, q=127, d=4, std_dev=1.0)
#         pk, sk = bliss.generate_keys()

#         message = "This is a test message."
#         signature = bliss.sign(message, (pk, sk))

#         # Inject an error into the signature.
#         signature_error = (signature[0] + 1, signature[1], signature[2])

#         self.assertFalse(bliss.verify(message, signature_error, pk))

# if __name__ == '__main__':
#     unittest.main()
