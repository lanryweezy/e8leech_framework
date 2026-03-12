import numpy as np
from core.lattices import LeechLattice

# This is a pre-flight check for CUDA/PyTorch availability
try:
    import torch
    HAS_TORCH = True
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False
    HAS_CUDA = False

class LeechGPU:
    """
    Experimental GPU-accelerated Leech Quantizer.
    If CUDA is available, this moves the 4,096-coset distance matrix
    to the GPU for massive parallel throughput.
    """
    def __init__(self):
        self.leech = LeechLattice()
        self.c2_cache = (2 * self.leech.golay.get_all_codewords()).astype(np.float32)
        
        if HAS_TORCH:
            self.device = torch.device("cuda" if HAS_CUDA else "cpu")
            self.c2_tensor = torch.from_numpy(self.c2_cache).to(self.device)
            print(f"LeechGPU: Using {self.device} for computation.")
        else:
            print("LeechGPU: PyTorch not found. Falling back to NumPy CPU.")

    def quantify_batch(self, X):
        if not HAS_TORCH:
            return self.leech.quantify_batch(X)
            
        N = X.shape[0]
        # Torch uses float32 for high-speed CUDA kernels
        X_tensor = torch.from_numpy(X.astype(np.float32)).to(self.device)
        
        # We still chunk to avoid VRAM overflow for massive batches
        chunk_size = 5000
        all_best_p = []
        
        for i in range(0, N, chunk_size):
            chunk = X_tensor[i:i+chunk_size]
            
            # (chunk_size, 1, 24) - (1, 4096, 24)
            # CUDA handles this broadcasting in parallel across thousands of cores
            diff = chunk.unsqueeze(1) - self.c2_tensor.unsqueeze(0)
            p_candidates = 2.0 * torch.round(diff / 4.0) * 2.0 + self.c2_tensor.unsqueeze(0)
            
            # Squared distance: (chunk_size, 4096)
            d_diff = chunk.unsqueeze(1) - p_candidates
            dists_sq = torch.sum(d_diff**2, dim=2)
            
            best_indices = torch.argmin(dists_sq, dim=1)
            all_best_p.append(p_candidates[torch.arange(len(chunk)), best_indices].cpu().numpy())
            
        return np.vstack(all_best_p)

if __name__ == "__main__":
    quantizer = LeechGPU()
    test_data = np.random.randn(1000, 24).astype(np.float32)
    start = time.time()
    res = quantizer.quantify_batch(test_data)
    print(f"Processed 1000 vectors in {time.time() - start:.4f}s")
