# Tutorial: Building Quantum-Safe Systems

This tutorial will guide you through the process of building a simple quantum-safe communication system using the e8leech library.

## 1. Key Exchange

The first step is to establish a shared secret between the two parties. We can use the LWE-based key exchange protocol for this.

```python
from e8leech.crypto.key_exchange import LWEKeyExchange

# Create two key exchange objects
alice = LWEKeyExchange(n=8, q=127)
bob = LWEKeyExchange(n=8, q=127)

# Get the public keys
alice_public_key = alice.get_public_key()
bob_public_key = bob.get_public_key()

# Generate the shared secrets
alice_shared_secret = alice.get_shared_secret(bob_public_key)
bob_shared_secret = bob.get_shared_secret(alice_public_key)

# The shared secrets should be the same
assert np.allclose(alice_shared_secret, bob_shared_secret)
```

## 2. Encryption and Decryption

Once a shared secret has been established, we can use it to encrypt and decrypt messages. We can use the LWE cryptosystem for this.

```python
from e8leech.crypto.lwe import LWE

# Create an LWE object with the shared secret
lwe = LWE(n=8, q=127)
lwe.secret_key = alice_shared_secret

# Encrypt a message
message = 1
ciphertext = lwe.encrypt(message)

# Decrypt the message
decrypted_message = lwe.decrypt(ciphertext)

assert message == decrypted_message
```

## 3. Digital Signatures

To ensure the authenticity of the messages, we can use a digital signature scheme. We can use the BLISS signature scheme for this.

```python
from e8leech.crypto.signatures import BLISS

# Create a BLISS object
bliss = BLISS(n=8, q=127, sigma=1.0)

# Sign a message
message = "This is a secret message."
signature = bliss.sign(message)

# Verify the signature
assert bliss.verify(message, signature)
```
