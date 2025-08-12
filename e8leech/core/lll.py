import numpy as np
from .gram_schmidt import gram_schmidt
from .vector import Vector

def lll(basis, d=0.75):

	basis = [Vector(v) for v in basis]

	gs = gram_schmidt(basis)

	def proj(u, v):
		return (u * v) / (v * v)

	k = 1
	while k < len(basis):

		for j in range(k - 1, -1, -1):

			mu = proj(basis[k], gs[j])

			if abs(mu) > 0.5:
				basis[k] = basis[k] - round(mu) * basis[j]
				gs = gram_schmidt(basis)

		if gs[k] * gs[k] >= (d - proj(basis[k], gs[k - 1]) ** 2) * (gs[k - 1] * gs[k - 1]):
			k = k + 1
		else:
			basis[k], basis[k - 1] = basis[k - 1], basis[k]
			gs = gram_schmidt(basis)
			k = max(k - 1, 1)

	return np.array([v.v for v in basis])
