import math
import collections
import numpy
import sys

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


class DTree:
	"""
	partition_x: Table with the instances selected for the partition (initially from the bootstrap).
	partition_y: numpy.ndarray. Classes of the instances of the partition.
	possible_attributes: List of integers, indexes of the possible attributes in x for the node.
	"""
	def __init__(self, partition_x, partition_y, random_state, possible_attributes):
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

		print(possible_attributes)
		self.attribute = selected_attribute

		self.children = dict()

		possible_attributes.remove(self.attribute)

		attribute_values = set(partition_x[:, self.attribute])

		for value in attribute_values:
			indexes = numpy.where(partition_x[:, self.attribute] == value)[0]
			x_with_value = partition_x[indexes]
			y_with_value = partition_y[indexes]
			child = DTree(x_with_value, y_with_value, random_state, possible_attributes)
