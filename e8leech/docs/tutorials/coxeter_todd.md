# Coxeter-Todd Lattice Tutorial

This tutorial shows how to use the `CoxeterToddLattice` class to perform some basic operations on the Coxeter-Todd lattice.

## Creating a Coxeter-Todd Lattice

To create a Coxeter-Todd lattice, you can use the `CoxeterToddLattice` class from the `e8leech.core.coxeter_todd` module:

```python
from e8leech.core.coxeter_todd import CoxeterToddLattice

ct = CoxeterToddLattice()
```

## Getting the Basis

To get the basis of the lattice, you can use the `basis` attribute:

```python
basis = ct.basis
print(basis)
```
