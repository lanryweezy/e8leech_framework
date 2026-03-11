import numpy as np
from core.lattices import E8Lattice

def test_e8_roots():
    e8 = E8Lattice()
    roots = e8.get_shortest_vectors()
    print(f"Number of E8 roots: {len(roots)}")
    
    # Verify norm squared is 2
    for r in roots:
        norm_sq = np.sum(r**2)
        if not np.isclose(norm_sq, 2.0):
            print(f"FAILED: Vector {r} has norm squared {norm_sq}")
            return
            
    print("SUCCESS: All roots have correct norm squared (2.0)")

if __name__ == "__main__":
    test_e8_roots()
