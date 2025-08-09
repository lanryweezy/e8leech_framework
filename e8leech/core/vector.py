from fractions import Fraction
import numpy as np

class Vector(object):

	def __init__(self, v):

		if type(v) == Vector:
			self.v = v.v
		elif type(v) == list or type(v) == tuple or type(v) == np.ndarray:
			self.v = v
		else:
			raise TypeError("Vector must be initialized with a list, tuple, Vector, or numpy array")

	def __len__(self):
		return len(self.v)

	def __getitem__(self, i):
		return self.v[i]

	def __repr__(self):
		return str(self.v)

	def __eq__(self, other):
		return self.v == other.v

	def __add__(self, other):

		v = []
		for i in range(len(self)):
			v.append(self[i] + other[i])

		return Vector(v)

	def __sub__(self, other):

		v = []
		for i in range(len(self)):
			v.append(self[i] - other[i])

		return Vector(v)

	def __mul__(self, other):

		if type(other) == Vector:
			dot = 0
			for i in range(len(self)):
				dot += self[i] * other[i]
			return dot
		else:
			v = []
			for i in range(len(self)):
				v.append(self[i] * other)
			return Vector(v)

	def __rmul__(self, other):
		return self * other

	def __truediv__(self, other):

		v = []
		for i in range(len(self)):
			v.append(self[i] / other)

		return Vector(v)

	def mag(self):

		total = 0
		for i in range(len(self)):
			total += self[i] ** 2

		return total ** 0.5

	def normalize(self):
		return self / self.mag()

	def proj(self, other):
		return ((self * other) / (other * other)) * other

	def fraction(self):

		v = []
		for i in range(len(self)):
			v.append(Fraction(self[i]).limit_denominator())

		return Vector(v)
