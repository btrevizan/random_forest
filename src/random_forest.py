import numpy
import src.utils as utils
from src.dtree import DTree

class RandomForest:
	def __init__(self, ntrees, random_state, table, classes):
		self.ntrees = ntrees
		self.random_state = random_state
		self.table = table
		self.classes = classes

		self.trees = []

		all_attributes = list(self.table.columns)

		for b in range(ntrees):
			bootstrap = list(utils.bootstrap(len(self.table), self.random_state))
			tree = DTree(bootstrap, self.table, self.classes, self.random_state, all_attributes)
			self.trees.append(tree)
