import numpy as np
from core.lattices import E8Lattice, GolayCode, LeechLattice

def test_golay():
    golay = GolayCode()
    codewords = golay.get_all_codewords()
    print(f"Golay Codewords: {len(codewords)}")
    weights = np.sum(codewords, axis=1)
    unique_weights = np.unique(weights)
    print(f"Unique Weights: {unique_weights}")
    # Extended Golay code weights should be 0, 8, 12, 16, 24
    expected_weights = [0, 8, 12, 16, 24]
    for w in expected_weights:
        count = np.sum(weights == w)
        print(f"Weight {w}: {count}")

def test_leech():
    print("Generating Leech minimal vectors (this might take a second)...")
    leech = LeechLattice()
    min_vecs = leech.generate_minimal_vectors()
    print(f"Total Minimal Vectors: {len(min_vecs)}")
    norms = np.sum(min_vecs**2, axis=1)
    print(f"Unique Norms^2: {np.unique(norms)}")

if __name__ == "__main__":
    test_golay()
    print("-" * 20)
    test_leech()
