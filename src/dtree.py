from collections import namedtuple
import src.utils as utils
from .model import Model
from copy import copy
import numpy as np
import math


Node = namedtuple('Node', ['id', 'info_gain', 'value', 'rule', 'children'])
"""Node representation as a named tuple.

:attr id: int
	Node's id number.
	
:attr info_gain: float
	Node's information gain.
	
:attr value: int
	Attribute's index or predicted class for leafs.

:attr rule: function
	Node's rule to map the attribute value in one of the children's key.

:attr children: dict
	Node's children, where: 
		- key: an attribute's value
		- value: Node
"""


class DecisionTree(Model):
	"""Represent a decision tree."""

	def __init__(self, random_state, numerical_attributes_indexes=[], attribute_values={}, benchmark=False):
		"""Initialize the decision tree. It does not fit the model. For that, you need to call fit(x, y).

		:param random_state: instance of numpy.random.RandomState
			Seed for random generator.

		:param numerical_attributes_indexes: list (default [])
			List of numeric attributes' index.

		:param attribute_values: dict {int: list} (optional)
			{attribute_index: attribute_unique_values}

		:param benchmark: bool (default False)
			Whether the decision tree is a benchmark or not.
		"""
		self.__tree = None
		self.__node_id = 0
		self.__benchmark = benchmark
		self.__random_state = random_state
		self.__attribute_values = attribute_values
		self.__numerical_attributes_indexes = numerical_attributes_indexes

	@property
	def node_id(self):
		id_number = self.__node_id
		self.__node_id += 1
		return id_number

	def is_leaf(self, node):
		return len(node.children) == 0

	def fit(self, x, y) -> None:
		available_attributes = list(range(x.shape[1]))
		self.__tree = self.__make_node(x, y, available_attributes)

	def predict(self, x) -> list:
		if not self.__tree:
			raise ValueError("You need to train the model first.")

		x = np.array(x)
		return [self.__find_leaf(x[i, :], self.__tree) for i in range(x.shape[0])]

	def __find_leaf(self, instance, node):
		if self.is_leaf(node):
			return node.value

		value = node.rule(instance[node.value])
		return self.__find_leaf(instance, node.children[value])

	def __make_node(self, x, y, available_attributes):
		sety, y_counts = np.unique(y, return_counts=True)

		# Subset has only one class, so stop recursion
		if len(sety) == 1:
			return Node(self.node_id, 1, sety[0], None, {})

		# There are no more attributes to choose from, so stop recursion
		if len(available_attributes) == 0:
			return Node(self.node_id, 1, utils.get_majority_class(y), None, {})

		# Cut the tree if the number of instances is below a threshold
		if y.size <= 50:
			print("Cut tree with threshold.")
			return Node(self.node_id, 1, utils.get_majority_class(y), None, {})

		# Cut the tree if the majority class has 75% of the instances
		for _, count in zip(sety, y_counts):
			if count / y.size >= 0.75:
				print("Cut tree by majority.")
				return Node(self.node_id, 1, utils.get_majority_class(y), None, {})

		# Select the best attribute and if it is numerical, get threshold
		attribute, threshold, info_gain = self.__select_attribute(x, y, available_attributes)
		new_available_attributes = copy(available_attributes)
		new_available_attributes.remove(attribute)

		# Select attribute's values and rule
		if attribute in self.__numerical_attributes_indexes:
			rule = self.__get_numerical_rule(threshold)
			attribute_values = np.zeros((x.shape[0],))
			attribute_values[rule(x[:, attribute])] = 1
			attribute_unique_values = [True, False]
		else:
			rule = self.__get_categorical_rule()
			attribute_values = x[:, attribute]
			attribute_unique_values = self.__attribute_values[attribute]

		children = {}

		for value in attribute_unique_values:
			attr_value_indexes = np.where(attribute_values == value)[0]

			if len(attr_value_indexes):
				children[value] = self.__make_node(x[attr_value_indexes, :], y[attr_value_indexes], new_available_attributes)
			else:
				children[value] = Node(self.node_id, 1, utils.get_majority_class(y), None, {})

		return Node(self.node_id, info_gain, attribute, rule, children)

	def __select_attribute(self, x, y, available_attributes):
		# Select √m attributes randomly
		if self.__benchmark:
			possible_attributes = available_attributes
		else:
			n_possible_attr = math.sqrt(len(available_attributes))
			possible_attributes = self.__random_state.choice(available_attributes, math.ceil(n_possible_attr), replace=False)

		class_entropy = utils.entropy(y)

		# Compute information gain for each attribute and get the maximum gain index
		gains = []
		thresholds = []
		for j in possible_attributes:
			if j in self.__numerical_attributes_indexes:
				gain, threshold = self.__numerical_information_gain(class_entropy, x[:, j], y)
			else:
				gain, threshold = utils.information_gain(class_entropy, x[:, j], y), None

			gains.append(gain)
			thresholds.append(threshold)

		max_gain_i = np.argmax(gains)
		return possible_attributes[max_gain_i], thresholds[max_gain_i], gains[max_gain_i]

	def __numerical_information_gain(self, class_entropy, x, y):
		threshold = None
		discretized_x = None
		max_gain = -math.inf

		thresholds = self.__get_thresholds(x, y)
		for t in thresholds:
			new_x = np.zeros(x.shape)
			new_x[x > t] = 1

			gain = utils.information_gain(class_entropy, new_x, y)
			if gain > max_gain:
				threshold = t
				max_gain = gain
				discretized_x = new_x

		return utils.information_gain(class_entropy, discretized_x, y), threshold

	def __get_thresholds(self, x, y):
		new_y = y[np.argsort(x)]
		return [(x[i - 1] + x[i]) / 2 for i in range(1, len(new_y)) if new_y[i - 1] != new_y[i]]

	def __get_categorical_rule(self, value=None):
		return lambda x: x

	def __get_numerical_rule(self, value):
		return lambda x: (x > value)

	def get_graph(self, dot, node=None, attr_names=None):
		if not node:
			node = self.__tree

		if len(node.children):
			if attr_names:
				label = "{}\nInfoGain: {}".format(attr_names[node.value], round(node.info_gain, 3))
			else:
				label = str(node.value)
		else:
			items = attr_names.items()
			items = sorted(items, key=lambda x: x[0])
			item = items[-1]
			label = "{} = {}".format(item[1], node.value)

		dot.node(str(node.id), label)

		for key, child in node.children.items():
			self.get_graph(dot, child, attr_names)
			dot.edge(str(node.id), str(child.id))
