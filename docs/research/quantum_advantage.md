# Quantum Advantage Analysis

The cryptographic schemes implemented in this library are based on the hardness of lattice problems, such as the Shortest Vector Problem (SVP) and the Closest Vector Problem (CVP). These problems are believed to be hard for both classical and quantum computers.

## Shor's Algorithm

Shor's algorithm can be used to efficiently solve integer factorization and the discrete logarithm problem, which are the foundations of many classical cryptographic schemes. However, Shor's algorithm is not known to be effective against lattice-based cryptography.

## Grover's Algorithm

Grover's algorithm can be used to speed up unstructured search problems. It can be used to speed up attacks against lattice-based cryptography, but the speedup is only quadratic. This means that to maintain the same level of security, the key sizes need to be doubled, which is a manageable overhead.
