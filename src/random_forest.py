import numpy
import src.utils as utils
from src.dtree import DTree

class RandomForest:
	def __init__(self, ntrees, random_state, x, y):
		self.ntrees = ntrees
		self.random_state = random_state
		self.x = x
		self.y = y

		self.trees = []

		all_attributes = [i for i in range(len(x[0]))]

		for b in range(ntrees):
			bootstrap = list(utils.bootstrap(len(self.x), self.random_state))
			tree = DTree(bootstrap, self.x, self.y, self.random_state, all_attributes)
			self.trees.append(tree)
