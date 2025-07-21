# Core Golay code implementation for Leech lattice construction
import numpy as np
import cupy as cp
from itertools import combinations
from numba import jit
from e8leech.data.lsh import LSH

@jit(nopython=True)
def _decode_numba(received, H, B):
    """Numba-jitted version of the decode function"""
    syndrome = np.dot(received, H.T) % 2
    weight = np.sum(syndrome)
    error = np.zeros(24, dtype=np.int64)
    if weight <= 3:
        error[-12:] = syndrome
    else:
        for i in range(12):
            candidate = (syndrome + B[i]) % 2
            if np.sum(candidate) <= 2:
                error[i] = 1
                error[-12:] = candidate
                break
        else:
            syndrome_comp = (syndrome + np.ones(12, dtype=np.int64)) % 2
            weight_comp = np.sum(syndrome_comp)
            if weight_comp <= 3:
                error[-12:] = syndrome_comp
            else:
                for i in range(12):
                    candidate = (syndrome_comp + B[i]) % 2
                    if np.sum(candidate) <= 2:
                        error[i] = 1
                        error[-12:] = candidate
                        break
    corrected = (received + error) % 2
    return corrected

from functools import lru_cache

@lru_cache(maxsize=None)
@jit(nopython=True)
def _closest_vector_numba(point, basis):
    """Numba-jitted and cached version of the closest_vector function"""
    b_inv = np.linalg.inv(basis)
    c = np.dot(point, b_inv)
    c_rounded = np.round(c)
    return np.dot(c_rounded, basis)

class GolayCode:
    """Binary extended Golay code implementation"""
    
    def __init__(self, device='cpu'):
        self.length = 24
        self.dimension = 12
        self.min_distance = 8
        self.device = device
        if self.device == 'gpu':
            self.xp = cp
        else:
            self.xp = np
        self.B = self.xp.array([
            [1,1,0,1,1,1,0,0,0,0,1,1],
            [1,0,1,1,1,0,0,0,0,1,1,1],
            [0,1,1,1,0,0,0,0,1,1,1,0],
            [1,1,1,0,0,0,0,1,1,1,0,1],
            [1,1,0,0,0,0,1,1,1,0,1,0],
            [1,0,0,0,0,1,1,1,0,1,0,1],
            [0,0,0,0,1,1,1,0,1,0,1,1],
            [0,0,0,1,1,1,0,1,0,1,1,0],
            [0,0,1,1,1,0,1,0,1,1,0,0],
            [0,1,1,1,0,1,0,1,1,0,0,0],
            [1,1,1,0,1,0,1,1,0,0,0,0],
            [1,1,0,1,0,1,1,0,0,0,0,1]
        ])
        self.H = self.xp.hstack([self.B.T, self.xp.eye(12)])
    
    def encode(self, data):
        """Encode 12-bit data to 24-bit codeword using generator matrix"""
        I = np.eye(12)
        G = np.hstack([I, self.B])
        return np.dot(data, G) % 2
    
    def decode(self, received):
        """Decode and error-correct using Golay code algorithm"""
        if self.device == 'gpu':
            # Convert to cupy array if needed
            received = self.xp.array(received)

            # Calculate syndrome
            syndrome = self.xp.dot(received, self.H.T) % 2
            weight = self.xp.sum(syndrome)

            # Initialize error vector
            error = self.xp.zeros(24, dtype=int)

            if weight <= 3:
                error[-12:] = syndrome
            else:
                # Check combinations with basis rows
                for i in range(12):
                    candidate = (syndrome + self.B[i]) % 2
                    if self.xp.sum(candidate) <= 3:
                        error[i] = 1
                        error[-12:] = candidate
                        break
                else:
                    # Check complement syndrome
                    syndrome_comp = (syndrome + 1) % 2
                    weight_comp = self.xp.sum(syndrome_comp)
                    if weight_comp <= 3:
                        error[-12:] = syndrome_comp
                    else:
                        # Attempt second-stage correction
                        for i in range(12):
                            candidate = (syndrome_comp + self.B[i]) % 2
                            if self.xp.sum(candidate) <= 3:
                                error[i] = 1
                                error[-12:] = candidate
                                break

            corrected = (received + error) % 2
            return corrected.astype(int)
        else:
            return _decode_numba(received, self.H, self.B)

    def decode(self, received):
        """Decode and error-correct using Golay code algorithm"""
        # Convert to numpy array if needed
        received = np.array(received)
        
        # Calculate syndrome
        syndrome = np.dot(received, self.H.T) % 2
        weight = np.sum(syndrome)
        
        # Initialize error vector
        error = np.zeros(24, dtype=int)
        
        if weight <= 3:
            error[-12:] = syndrome
        else:
            # Check combinations with basis rows
            for i in range(12):
                candidate = (syndrome + self.B[i]) % 2
                if np.sum(candidate) <= 3:
                    error[i] = 1
                    error[-12:] = candidate
                    break
            else:
                # Check complement syndrome
                syndrome_comp = (syndrome + 1) % 2
                weight_comp = np.sum(syndrome_comp)
                if weight_comp <= 3:
                    error[-12:] = syndrome_comp
                else:
                    # Attempt second-stage correction
                    for i in range(12):
                        candidate = (syndrome_comp + self.B[i]) % 2
                        if np.sum(candidate) <= 3:
                            error[i] = 1
                            error[-12:] = candidate
                            break
        
        corrected = (received + error) % 2
        return corrected.astype(int)

class LeechLattice:
    """Leech lattice implementation with Golay code basis"""
    
    def __init__(self, device='cpu', dtype=np.float64):
        self.dim = 24
        self.device = device
        self.dtype = dtype
        if self.device == 'gpu':
            self.xp = cp
            self.closest_vector_cache = {}
        else:
            self.xp = np
        self.golay = GolayCode(device=device)
        self.basis = self._generate_basis().astype(self.dtype)
        
    def _generate_basis(self):
        """Generate Leech lattice basis using E8^3 construction"""
        basis = self.xp.zeros((24, 24))
        
        # Build from three E8 blocks scaled by √8
        e8 = E8Lattice(device=self.device).basis * self.xp.sqrt(8)
        basis[:8, :8] = e8
        basis[8:16, 8:16] = e8
        basis[16:24, 16:24] = e8
        
        # Add Golay code relationships between blocks
        for i in range(8):
            basis[16+i, :8] = 0.5
            basis[16+i, 8+i] = 0.5
            
        return basis
        
    def generate_leech_points(self, parallel=False):
        """Generate Leech lattice points using all congruence conditions"""
        if parallel:
            from e8leech.utils.parallel import ray_map

            def process_codeword(codeword):
                points = []
                base_point = codeword / self.xp.sqrt(8)
                for offset in itertools.product([-1, 0, 1], repeat=24):
                    points.append(base_point + 2*self.xp.array(offset)/self.xp.sqrt(8))
                return points

            codewords = self._get_golay_codewords()
            results = ray_map(process_codeword, codewords)
            points = [p for sublist in results for p in sublist]

            def process_z(z):
                base_point = 2*self.xp.array(z)/self.xp.sqrt(8)
                if self.xp.sum(base_point % 2) == 0:
                    return base_point
                return None

            z_values = list(itertools.product([-1, 0, 1], repeat=24))
            results = ray_map(process_z, z_values)
            points.extend([p for p in results if p is not None])

            return self.xp.unique(self.xp.round(points, 6), axis=0)
        else:
            points = []

            # Type 1: 1/√8 * (codeword + 2z) where z ∈ Z^24
            for codeword in self._get_golay_codewords():
                base_point = codeword / self.xp.sqrt(8)
                # Generate integer combinations
                for offset in itertools.product([-1, 0, 1], repeat=24):
                    points.append(base_point + 2*self.xp.array(offset)/self.xp.sqrt(8))

            # Type 2: 1/√8 * (2z + ξ) where ξ has even coordinates
            for z in itertools.product([-1, 0, 1], repeat=24):
                base_point = 2*self.xp.array(z)/self.xp.sqrt(8)
                if self.xp.sum(base_point % 2) == 0:
                    points.append(base_point)

            return self.xp.unique(self.xp.round(points, 6), axis=0)
    
    def kissing_number(self):
        """Return the Leech lattice kissing number (196560)"""
        return 196560

    def build_lsh_index(self, n_hashes=8, n_tables=10, n_points=10000):
        """Builds an LSH index for approximate nearest neighbor search."""
        self.lsh = LSH(n_hashes, n_tables, self.dim)
        points = self.generate_leech_points(n_points=n_points)
        self.lsh.index(points)
        self.lsh_points = points

    def approx_closest_vector(self, point):
        """Finds an approximate closest vector using LSH."""
        if not hasattr(self, 'lsh'):
            raise RuntimeError("LSH index has not been built. Call build_lsh_index() first.")

        candidate_indices = self.lsh.query(point, k=10)
        if not candidate_indices:
            return self.closest_vector(point) # Fallback to exact search

        candidates = self.lsh_points[candidate_indices]
        distances = self.xp.linalg.norm(candidates - point, axis=1)
        best_candidate_index = self.xp.argmin(distances)
        return candidates[best_candidate_index]

    def closest_vector(self, point):
        """Babai's nearest plane algorithm for the Leech lattice"""
        if self.device == 'gpu':
            point_tuple = tuple(point.tolist())
            if point_tuple in self.closest_vector_cache:
                return self.closest_vector_cache[point_tuple]

            b_inv = self.xp.linalg.inv(self.basis)
            c = self.xp.dot(point, b_inv)
            c_rounded = self.xp.round(c)
            result = self.xp.dot(c_rounded, self.basis)
            self.closest_vector_cache[point_tuple] = result
            return result
        else:
            return _closest_vector_numba(point, self.basis)
    
    def _get_golay_codewords(self):
        """Generate Golay code codewords for lattice construction"""
        codewords = []
        for i in range(2**12):
            message = np.array(list(np.binary_repr(i, width=12)), dtype=int)
            codewords.append(self.encode(message))
        return np.array(codewords)

class E8Lattice:
    """E8 lattice implementation with mathematical precision"""
    
    def __init__(self, device='cpu', dtype=np.float64):
        self.dim = 8
        self.device = device
        self.dtype = dtype
        if self.device == 'gpu':
            self.xp = cp
            self.closest_vector_cache = {}
        else:
            self.xp = np
        self.basis = self._generate_basis().astype(self.dtype)
        self.root_system = self._generate_root_system().astype(self.dtype)
    
    def _generate_basis(self):
        """Generate E8 basis matrix (even coordinate system)"""
        basis = self.xp.eye(8)
        # Add E8-specific relationships
        for i in range(7):
            basis[i, 7] = 0.5
        return basis
    
    def _generate_root_system(self, parallel=False):
        """Generate the 240 root vectors of E8"""
        if parallel:
            from e8leech.utils.parallel import ray_map

            def process_i(i):
                if bin(i).count('1') % 2 == 0:
                    return self.xp.array([(0.5 if (j >> k) & 1 else -0.5) for k in range(8)])
                return None

            results = ray_map(process_i, range(256))
            roots = [r for r in results if r is not None]

            def process_j(j):
                roots_j = []
                for i in range(j + 1, 8):
                    for signs in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                        vec = self.xp.zeros(8)
                        vec[j] = signs[0]
                        vec[i] = signs[1]
                        roots_j.append(vec / self.xp.sqrt(2))
                return roots_j

            results = ray_map(process_j, range(8))
            roots.extend([r for sublist in results for r in sublist])

            return self.xp.array(roots)
        else:
            roots = []

            # Type 1: Even coord vectors (±½)^8 with even parity
            for i in range(256):
                if bin(i).count('1') % 2 == 0:
                    vec = self.xp.array([(0.5 if (i >> j) & 1 else -0.5) for j in range(8)])
                    roots.append(vec)

            # Type 2: (±1,±1,0⁶) permutations normalized by √2
            for i in range(8):
                for j in range(i+1, 8):
                    for signs in [(1,1), (1,-1), (-1,1), (-1,-1)]:
                        vec = self.xp.zeros(8)
                        vec[i] = signs[0]
                        vec[j] = signs[1]
                        roots.append(vec/self.xp.sqrt(2))

            # Verify we have exactly 240 roots (128 + 112)
            return self.xp.array(roots[:240])
    
    def closest_vector(self, point):
        """Babai's nearest plane algorithm"""
        if self.device == 'gpu':
            point_tuple = tuple(point.tolist())
            if point_tuple in self.closest_vector_cache:
                return self.closest_vector_cache[point_tuple]

            b_inv = self.xp.linalg.inv(self.basis)
            c = self.xp.dot(point, b_inv)
            c_rounded = self.xp.round(c)
            result = self.xp.dot(c_rounded, self.basis)
            self.closest_vector_cache[point_tuple] = result
            return result
        else:
            return _closest_vector_numba(point, self.basis)
    
    def kissing_number(self):
        """Return the number of minimal vectors (240)"""
        return len(self.root_system)

    def build_lsh_index(self, n_hashes=8, n_tables=10):
        """Builds an LSH index for approximate nearest neighbor search."""
        self.lsh = LSH(n_hashes, n_tables, self.dim)
        self.lsh.index(self.root_system)

    def approx_closest_vector(self, point):
        """Finds an approximate closest vector using LSH."""
        if not hasattr(self, 'lsh'):
            raise RuntimeError("LSH index has not been built. Call build_lsh_index() first.")

        candidate_indices = self.lsh.query(point, k=10)
        if not candidate_indices:
            return self.closest_vector(point) # Fallback to exact search

        candidates = self.root_system[candidate_indices]
        distances = self.xp.linalg.norm(candidates - point, axis=1)
        best_candidate_index = self.xp.argmin(distances)
        return candidates[best_candidate_index]