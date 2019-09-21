from src.dtree import DecisionTree
import src.utils as utils
from .model import Model
from copy import copy


class RandomForest(Model):
	"""Random forest representation."""

	def __init__(self, ntrees, random_state, numerical_attributes_indexes=[]):
		self.__trees = []
		self.__ntrees = ntrees
		self.__random_state = random_state
		self.__numerical_attributes_indexes = numerical_attributes_indexes

	@property
	def trees(self):
		return copy(self.__trees)

	def fit(self, x, y) -> None:

		for i in range(self.__ntrees):
			bootstrap = utils.bootstrap(x.shape[0], self.__random_state)

			tree = DecisionTree(self.__random_state, self.__numerical_attributes_indexes)
			tree.fit(x[bootstrap, :], y[bootstrap])

			self.__trees.append(tree)

	def predict(self, x) -> list:
		if len(self.__trees) == 0:
			raise ValueError("You need to train the model first.")

		y_pred = [self.__trees[i].predict(x) for i in range(len(self.__trees))]
		y_pred = zip(*y_pred)

		return [utils.get_majority_class(list(predictions)) for predictions in y_pred]




