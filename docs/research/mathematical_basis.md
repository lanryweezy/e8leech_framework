# Mathematical Basis

## The E8 Lattice

The E8 lattice is a remarkable 8-dimensional lattice. Its points can be described as the set of vectors $x \in \mathbb{R}^8$ such that:
1. The coordinates of $x$ are either all integers or all half-integers.
2. The sum of the coordinates of $x$ is an even integer.

The norm of a vector in E8 is always an even integer. The number of vectors of a given norm can be calculated using the theta function of the lattice:
$$ \theta_{E8}(\tau) = \sum_{x \in E8} q^{\|x\|^2/2} = 1 + 240q + 2160q^2 + 6720q^3 + \dots $$
where $q = e^{2\pi i \tau}$.

For more details, see the original paper by Conway and Sloane: [Sphere Packings, Lattices and Groups](https://arxiv.org/abs/math/0012029).

## The Leech Lattice

The Leech lattice is a 24-dimensional lattice that is even more remarkable than E8. It is the unique even, unimodular lattice in 24 dimensions with no roots (i.e., no vectors of norm 2). Its points can be constructed using the Golay code.
