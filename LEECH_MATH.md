# Leech Lattice Notes

## Minimal Vectors Construction
The Leech lattice has 196,560 minimal vectors of norm 4. They fall into three shapes:

1. **Shape 1: (4, 4, 0^22)**
   - Vectors of the form $(\pm 4, \pm 4, 0, \dots, 0)$ where there are two non-zero coordinates.
   - Count: $2^2 \times \binom{24}{2} = 4 \times 276 = 1,104$.

2. **Shape 2: (2^8, 0^16)**
   - Support is an octad of the Binary Golay Code (G24).
   - Coordinates are $\pm 2$ at the support positions.
   - Sum of coordinates must be $\equiv 0 \pmod{4}$.
   - Count: $759$ octads $\times 2^7$ (sign combinations with even sum) $= 97,152$.

3. **Shape 3: (3, 1^23)**
   - Vectors of the form $(\mp 3, \pm 1, \pm 1, \dots, \pm 1)$.
   - The $-3$ can be at any of the 24 positions.
   - The pattern of $\pm 1$ must relate to the Golay code words.
   - Count: $24 \times 2^{12} = 98,304$.

**Total:** $1,104 + 97,152 + 98,304 = 196,560$.

## AI Application
By mapping data points to these minimal vectors, we can quantize high-dimensional AI embeddings into a discrete, highly-symmetrical space. This effectively turns an "unstructured" vector into a "structured" lattice address.
