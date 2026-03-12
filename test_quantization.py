import numpy as np
from core.lattices import E8Lattice

def test_e8_quantization():
    e8 = E8Lattice()
    
    # Test 1: Vector already in lattice
    v1 = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    q1 = e8.quantify(v1)
    print(f"Input: {v1}, Quantified: {q1}, Match: {np.allclose(v1, q1)}")
    
    # Test 2: Random vector
    np.random.seed(42)
    v2 = np.random.randn(8) * 2
    q2 = e8.quantify(v2)
    print(f"Input: {v2}")
    print(f"Quantified: {q2}")
    print(f"Is in lattice: {e8.is_in_lattice(q2)}")
    
    # Test 3: Half-integer vector
    v3 = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    q3 = e8.quantify(v3)
    print(f"Input: {v3}, Quantified: {q3}, Match: {np.allclose(v3, q3)}")

if __name__ == "__main__":
    test_e8_quantization()
