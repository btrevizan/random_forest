import math

# Attribute types:
#   float64
#   int64
#   float32
#   int8
#   bool
#   object (string)

def compute_information_gain(attribute, table, classes):
	pass


class DTree:
	def __init__(self, table, classes, random_state, attributes):
		numero_atributos_selecionar = math.ceil(math.sqrt(len(attributes)))
		atributos_possiveis = list(random_state.choice(attributes, numero_atributos_selecionar, replace = False))

		melhor_ganho_info = -math.inf

		atributo_escolhido = None

		for atributo in atributos_possiveis:
			ganho_info = compute_information_gain(atributo, table, classes)
			if (ganho_info > melhor_ganho_info):
				atributo_escolhido = atributo
				melhor_ganho_info = ganho_info

		self.attribute = atributo_escolhido



