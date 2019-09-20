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


# attribute index : sorted split points.
numeric_attributes_split_points = dict()

# attribute index: possible values.
categoric_attribute_values = dict()

all_attribute_names = []


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


def attribute_entropy(attribute, partition_x, partition_y, is_numeric):
	"""
	attribute: integer. Index of the attribute in partition_x to compute entropy for information gain.
	partition_x: Instances from the table.
	partition_y: Classes of the instances from the table.
	is_numeric: Boolean. Information if the attribute is numeric.
	"""
	if not is_numeric:
		attribute_values = list(set(partition_x[:, attribute]))
	else:
		attribute_values = numeric_attributes_split_points[attribute]
	partition_size = len(partition_x)
	entropy = 0.0
	for i in range(len(attribute_values)):
		if is_numeric and attribute_values[i] == math.inf:
			continue  # skip inf.

		if not is_numeric:
			indexes = numpy.where(partition_x[:, attribute] == attribute_values[i])[0]
		else:
			indexes = numpy.where( (partition_x[:, attribute] > attribute_values[i]) & (partition_x[:, attribute] <= attribute_values[i+1]) )[0]

		if len(indexes) == 0:
			continue

		instances_with_value = partition_x[indexes]
		weight = len(instances_with_value) / partition_size
		entropy += weight * class_entropy(partition_y[indexes])

	return entropy


def get_majoritary_class(y):
	"""
	numpy.ndarray. Classes of the instances of the partition.
	"""
	return utils.majority_voting(y)


node_id = 0  # ids of the nodes

class DTree:
	"""
	partition_x: Table with the instances selected for the partition (initially from the bootstrap).
	partition_y: numpy.ndarray. Classes of the instances of the partition.
	possible_attributes: List of integers, indexes of the possible attributes in x for the node.
	numeric_attributes_indexes: List of integers, the indexes of the numeric attributes of the data.
	"""
	def __init__(self, partition_x, partition_y, random_state, possible_attributes, numeric_attributes_indexes):
		global node_id
		self.id = str(node_id)
		node_id += 1

		self.type = "intermediate"   # type of node: intermediate or leaf.
		self.predicted_class = get_majoritary_class(partition_y)  # if the node is a leaf: the predicted class.

		self.attribute = None  # if the node is intermediate: the attribute associated.
		self.children = None   # if the node is intermediate: children of the node.

		self.number_of_nodes = 1  # number of nodes in the tree from this node.

		self.attribute_type = None  # numeric or categoric.

		# check for stop conditions:
		# 1: pure node:
		number_of_classes_in_partition = len(set(partition_y))
		if (number_of_classes_in_partition == 1):
			self.type = "leaf"
			self.predicted_class = partition_y[0]
		# 2: possible_attributes is empty:
		elif len(possible_attributes) == 0:
			self.type = "leaf"

		# compute attribute for the node:
		else :

			# select attributes possible to choose:
			number_attributes_choose = math.ceil(math.sqrt(len(possible_attributes)))
			attributes_choose = list(random_state.choice(possible_attributes, number_attributes_choose, replace = False))

			# compute information gain of all attributes selected:
			best_info_gain = -math.inf
			selected_attribute = None

			partition_entropy = class_entropy(partition_y)

			for attribute in attributes_choose:
				info_gain = partition_entropy - attribute_entropy(attribute, partition_x, partition_y, (attribute in numeric_attributes_indexes))
				if (info_gain > best_info_gain):
					selected_attribute = attribute
					best_info_gain = info_gain

			self.attribute = selected_attribute
			self.children = dict()

			# allow attribute repetition in different branches of the tree:
			new_possible_attributes = copy.copy(possible_attributes)
			new_possible_attributes.remove(self.attribute)
			# 

			if self.attribute in numeric_attributes_indexes:
				self.attribute_type = "numeric"
				attribute_values = numeric_attributes_split_points[self.attribute]
			else:
				self.attribute_type = "categoric"
				attribute_values = list(set(partition_x[:, self.attribute]))

			# create one child for each value:
			for i in range(len(attribute_values)):

				if self.attribute_type == "categoric":
					indexes = numpy.where(partition_x[:, self.attribute] == attribute_values[i])[0]
					value = attribute_values[i]
				elif self.attribute_type == "numeric":
					if attribute_values[i] == math.inf:
						continue
					indexes = numpy.where( (partition_x[:, self.attribute] > attribute_values[i]) & (partition_x[:, self.attribute] <= attribute_values[i+1]) )[0]
					value = (attribute_values[i], attribute_values[i+1])  # numeric attribute: value will be range.

				x_with_value = partition_x[indexes]
				y_with_value = partition_y[indexes]

				# check for stop condition: one of the resulting partitions is empty:
				if len(x_with_value) == 0:
					self.type = "leaf"
					self.children = None
					self.number_of_nodes = 1
					break

				child = DTree(x_with_value, y_with_value, random_state, new_possible_attributes, numeric_attributes_indexes)
				self.number_of_nodes += child.number_of_nodes
				self.children[value] = child

	def predict(self, instance):
		"""
		instance: List with the values of the attributes, in order.
		"""
		if self.type == "leaf":
			return self.predicted_class
		else:
			inst_attribute_value = instance[self.attribute]
			if self.attribute_type == "categoric":
				if not inst_attribute_value in self.children:  # value of attribute not present in children:
					return self.predicted_class  # intermediate predicted class.
				else:
					child = self.children[inst_attribute_value]
					return child.predict(instance)
			else:
				for k, v in self.children.items():
					if (inst_attribute_value > k[0]) and (inst_attribute_value <= k[1]):
						return v.predict(instance)


	def get_graph(self, dot):
		if self.type == "leaf":
			label = str(self.predicted_class)
		elif self.type == "intermediate":
			label = all_attribute_names[self.attribute]
		dot.node(self.id, label)
		if self.type == "intermediate":
			for k, v in self.children.items():
				v.get_graph(dot)
				dot.edge(self.id, v.id, label=str(k))
