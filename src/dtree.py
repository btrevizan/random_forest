import math
import collections
import copy
import numpy
import sys
import src.utils as utils

# Attribute types:
#   float64
#   int64
#   float32
#   int8
#   bool
#   object (string)


def class_entropy(y):
	""" 
	y: Class vector.
	"""
	total_instances = len(y)
	entropy = 0.0
	for k, v in collections.Counter(y).items():
		pk = v / total_instances
		entropy += pk * numpy.log2(pk)
	entropy *= -1
	return entropy


def attribute_entropy(attribute, partition_x, partition_y):
	"""
	attribute: integer. Index of the attribute in partition_x to compute entropy for information gain.
	partition_x: Instances from the table.
	partition_y: Classes of the instances from the table.
	"""
	attribute_values = set(partition_x[:, attribute])
	partition_size = len(partition_x)
	entropy = 0.0
	for value in attribute_values:
		indexes = numpy.where(partition_x[:, attribute] == value)[0]
		instances_with_value = partition_x[indexes]
		weight = len(instances_with_value) / partition_size
		entropy += weight * class_entropy(partition_y[indexes])

	return entropy


def get_majoritary_class(y):
	"""
	numpy.ndarray. Classes of the instances of the partition.
	"""
	return utils.majority_voting(y)


class DTree:
	"""
	partition_x: Table with the instances selected for the partition (initially from the bootstrap).
	partition_y: numpy.ndarray. Classes of the instances of the partition.
	possible_attributes: List of integers, indexes of the possible attributes in x for the node.
	"""
	def __init__(self, partition_x, partition_y, random_state, possible_attributes):
		self.type = "intermediate"   # type of node: intermediate or leaf.
		self.predicted_class = None  # if the node is a leaf: the predicted class.

		self.attribute = None  # if the node is intermediate: the attribute associated.
		self.children = None   # if the node is intermediate: children of the node.

		# check for stop conditions:
		# 1: pure node:
		number_of_classes_in_partition = len(set(partition_y))
		if (number_of_classes_in_partition == 1):
			self.type = "leaf"
			self.predicted_class = partition_y[0]
		# 2: possible_attributes is empty:
		elif len(possible_attributes) == 0:
			self.type = "leaf"
			self.predicted_class = get_majoritary_class(partition_y)

		# compute attribute for the node:
		else :

			number_attributes_choose = math.ceil(math.sqrt(len(possible_attributes)))
			attributes_choose = list(random_state.choice(possible_attributes, number_attributes_choose, replace = False))

			best_info_gain = -math.inf
			selected_attribute = None

			partition_entropy = class_entropy(partition_y)

			for attribute in attributes_choose:
				info_gain = partition_entropy - attribute_entropy(attribute, partition_x, partition_y)
				if (info_gain > best_info_gain):
					selected_attribute = attribute
					best_info_gain = info_gain

			self.attribute = selected_attribute
			self.children = dict()

			# allow attribute repetition in different branches of the tree:
			new_possible_attributes = copy.copy(possible_attributes)
			new_possible_attributes.remove(self.attribute)
			# 

			attribute_values = set(partition_x[:, self.attribute])

			for value in attribute_values:
				indexes = numpy.where(partition_x[:, self.attribute] == value)[0]
				x_with_value = partition_x[indexes]
				y_with_value = partition_y[indexes]

				# check for stop condition: one of the resulting partitions is empty:
				if len(x_with_value) == 0:
					self.type = "leaf"
					self.predicted_class = get_majoritary_class(partition_y)
					self.children = None
					break

				child = DTree(x_with_value, y_with_value, random_state, new_possible_attributes)
				self.children[value] = child
