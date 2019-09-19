import numpy
import src.utils as utils
from src.dtree import DTree

class RandomForest:
	def __init__(self, ntrees, random_state, x, y, numerical_attributes):
		self.ntrees = ntrees
		self.random_state = random_state
		self.x = x
		self.y = y

		self.trees = []

		all_attributes = [i for i in range(len(x[0]))]

		numeric_attributes = list(zip(*numerical_attributes))
		numeric_attributes_indexes = tuple([int(x) for x in numeric_attributes[1]])
		numeric_attributes_names = numeric_attributes[0]

		for b in range(ntrees):
			bootstrap = list(utils.bootstrap(len(self.x), self.random_state))
			x_bootstrap_partition = self.x[bootstrap]
			y_bootstrap_partition = self.y[bootstrap]
			tree = DTree(x_bootstrap_partition, y_bootstrap_partition, self.random_state, all_attributes, numeric_attributes_indexes, numeric_attributes_names)
			self.trees.append(tree)

