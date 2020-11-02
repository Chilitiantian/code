import numpy as np

class Piecewise_constant(object):
	def __init__(self, boundaries, values):
		self.boundaries = boundaries
		self.values = values

	def updata(self, step):	
		idx = [i for i, boundarie in enumerate(self.boundaries) if step < boundarie]
		if not idx:
			return self.values[-1]
		else:
			return self.values[idx[0]]