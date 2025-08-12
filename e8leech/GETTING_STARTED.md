# Getting Started

### Installation

To install the E8Leech framework, you will need to have Python 3.6 or later installed. You can then install the framework using pip:

```
pip install e8leech
```

### Basic Usage

Here is an example of how to use the framework to create an E8 lattice and find the closest lattice point to a given vector:

```python
from e8leech.core.e8_lattice import E8Lattice
import numpy as np

# Create an E8 lattice
e8 = E8Lattice()

# Create a random vector
v = np.random.rand(8)

# Find the closest lattice point
closest_point = e8.quantize(v)

print("Closest point:", closest_point)
```
