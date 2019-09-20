from .model import Model
import math
import src.utils as utils
from src.dtree import DTree, numeric_attributes_split_points

def compute_numeric_attribues_split_points(values_and_classes, attribute_index):
	# [0]: value.  [1]: class.
	values_and_classes.sort(key=lambda x: x[0])
	all_split_points = set()
	for i in range(1, len(values_and_classes)):
		if values_and_classes[i][1] != values_and_classes[i-1][1]:  # two consecutive different classes.
			split_point = (values_and_classes[i][0] + values_and_classes[i-1][0]) / 2
			all_split_points.add(split_point)
	sorted_split_points = sorted(list(all_split_points))
	sorted_split_points.insert(0, -math.inf)
	sorted_split_points.append(math.inf)
	numeric_attributes_split_points[attribute_index] = sorted_split_points



class RandomForest(Model):
	def __init__(self, ntrees, random_state, x, y, numerical_attributes):
		self.ntrees = ntrees
		self.random_state = random_state
		self.x = x
		self.y = y

		self.trees = []

		all_attributes = [i for i in range(len(x[0]))]

		numeric_attributes = list(zip(*numerical_attributes))
		numeric_attributes_indexes = [int(x) for x in numeric_attributes[1]]
		
		# treat numeric attributes:
		for i in numeric_attributes_indexes:
			values = x[:, i]
			values_and_classes = list(zip(values, y))

			compute_numeric_attribues_split_points(values_and_classes, i)

		#

		for b in range(ntrees):
			bootstrap = list(utils.bootstrap(len(self.x), self.random_state))
			x_bootstrap_partition = self.x[bootstrap]
			y_bootstrap_partition = self.y[bootstrap]
			tree = DTree(x_bootstrap_partition, y_bootstrap_partition, self.random_state, all_attributes, numeric_attributes_indexes)
			self.trees.append(tree)

