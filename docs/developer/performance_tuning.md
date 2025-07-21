# Performance Tuning Guide

This guide provides some tips on how to improve the performance of the e8leech library.

## 1. Use GPU Acceleration

If you have an NVIDIA GPU, you can use the `device='gpu'` parameter when creating the `E8Lattice` and `LeechLattice` objects to accelerate the computations.

```python
from e8leech.core.e8_lattice import E8Lattice

e8_gpu = E8Lattice(device='gpu')
```

## 2. Use Numba JIT Compilation

For CPU-bound functions, you can use Numba's JIT compilation to improve performance. The `closest_vector` method is already JIT-compiled for CPU execution.

## 3. Use Parallel Processing

For large computations, you can use the `parallel=True` parameter in the `generate_leech_points` and `_generate_root_system` methods to distribute the computation across multiple cores.

```python
from e8leech.core.leech_lattice import LeechLattice

leech = LeechLattice()
points = leech.generate_leech_points(parallel=True)
```

## 4. Use Memory-Efficient Representations

If you are working with large lattices, you can use the `dtype=np.float16` parameter to reduce the memory footprint.

```python
from e8leech.core.e8_lattice import E8Lattice
import numpy as np

e8 = E8Lattice(dtype=np.float16)
```

## 5. Use Approximate Nearest Neighbor Search

For applications where an approximate closest vector is sufficient, you can use the `approx_closest_vector` method, which uses Locality-Sensitive Hashing (LSH) to find an approximate solution much faster than the exact method.

```python
from e8leech.core.e8_lattice import E8Lattice

e8 = E8Lattice()
e8.build_lsh_index()
point = np.random.randn(8)
closest = e8.approx_closest_vector(point)
```
