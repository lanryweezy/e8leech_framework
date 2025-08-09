from .vector import Vector

def gram_schmidt(*args, **kwargs):

	if len(args) == 1 and type(args[0]) == list:
		vectors = args[0]
	else:
		vectors = list(args)

	for i in range(len(vectors)):
		vectors[i] = Vector(vectors[i])

	gs = []
	for i in range(len(vectors)):

		v = vectors[i]

		p = Vector([0 for _ in range(len(v))])
		for j in range(len(gs)):
			p = p + v.proj(gs[j])

		gs.append(v - p)

	if "normalize" in kwargs and kwargs["normalize"] == True:
		for i in range(len(gs)):
			gs[i] = gs[i].normalize()

	if "fraction" in kwargs and kwargs["fraction"] == True:
		for i in range(len(gs)):
			gs[i] = gs[i].fraction()

	return gs
