# Leech Lattice Tutorial

This tutorial shows how to use the `LeechLattice` class to perform some basic operations on the Leech lattice.

## Creating a Leech Lattice

To create a Leech lattice, you can use the `LeechLattice` class from the `e8leech.core.leech_lattice` module:

```python
from e8leech.core.leech_lattice import LeechLattice

leech = LeechLattice()
```

## Finding the Closest Lattice Point

To find the closest lattice point to a given vector, you can use the `quantize` method:

```python
import numpy as np

v = np.random.rand(24)
closest_point = leech.quantize(v)

print("Closest point:", closest_point)
```

## Checking if a Vector is in the Lattice

To check if a vector is in the lattice, you can use the `is_valid` method:

```python
v = np.array([-2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) / np.sqrt(8)
print(leech.is_valid(v))
```

## Getting the Root System

To get the root system of the lattice, you can use the `get_root_system` method:

```python
root_system = leech.get_root_system()
print(root_system)
```
