import math

# Attribute types:
#   float64
#   int64
#   float32
#   int8
#   bool
#   object (string)

def compute_information_gain(attribute, x, y):
	return 1


class DTree:
	def __init__(self, x, y, random_state, possible_attributes):
		number_attributes_choose = math.ceil(math.sqrt(len(possible_attributes)))
		attributes_choose = list(random_state.choice(possible_attributes, number_attributes_choose, replace = False))

		best_info_gain = -math.inf

		selected_attribute = None

		for attribute in attributes_choose:
			info_gain = compute_information_gain(attribute, x, y)
			if (info_gain > best_info_gain):
				selected_attribute = attribute
				best_info_gain = info_gain

		self.attribute = selected_attribute



