# E8 Lattice Tutorial

This tutorial shows how to use the `E8Lattice` class to perform some basic operations on the E8 lattice.

## Creating an E8 Lattice

To create an E8 lattice, you can use the `E8Lattice` class from the `e8leech.core.e8_lattice` module:

```python
from e8leech.core.e8_lattice import E8Lattice

e8 = E8Lattice()
```

## Finding the Closest Lattice Point

To find the closest lattice point to a given vector, you can use the `quantize` method:

```python
import numpy as np

v = np.random.rand(8)
closest_point = e8.quantize(v)

print("Closest point:", closest_point)
```

## Checking if a Vector is in the Lattice

To check if a vector is in the lattice, you can use the `is_valid` method:

```python
v = np.array([1, 1, 0, 0, 0, 0, 0, 0])
print(e8.is_valid(v))
```

## Getting the Root System

To get the root system of the lattice, you can use the `get_root_system` method:

```python
root_system = e8.get_root_system()
print(root_system)
```
