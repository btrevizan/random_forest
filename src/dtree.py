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



def compute_information_gain(attribute, x, y):
	"""
	attribute: integer. Index of the attribute in x to compute information gain.
	"""


def compute_entropy(y):
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

class DTree:
	"""
	bootstrap: List of integers, indexes of the instances selected from x.
	possible_attributes: List of integers, indexes of the possible attributes in x for the node.
	"""
	def __init__(self, bootstrap, x, y, random_state, possible_attributes):
		number_attributes_choose = math.ceil(math.sqrt(len(possible_attributes)))
		attributes_choose = list(random_state.choice(possible_attributes, number_attributes_choose, replace = False))

		best_info_gain = -math.inf
		selected_attribute = None		

		entropy = compute_entropy(y)

		for attribute in attributes_choose:
			info_gain = compute_information_gain(attribute, x, y)
			if (info_gain > best_info_gain):
				selected_attribute = attribute
				best_info_gain = info_gain

		self.attribute = selected_attribute



