from src.dtree import DecisionTree
from threading import Thread
import src.utils as utils
from .model import Model
from copy import copy
import os


class RandomForest(Model):
	"""Random forest representation."""

	def __init__(self, ntrees, random_state, numerical_attributes_indexes=[]):
		self.__ntrees = ntrees
		self.__random_state = random_state
		self.__numerical_attributes_indexes = numerical_attributes_indexes
		self.__trees = [DecisionTree(self.__random_state, self.__numerical_attributes_indexes) for _ in range(self.__ntrees)]

	@property
	def trees(self):
		return copy(self.__trees)

	def fit(self, x, y) -> None:
		
		for i in range(self.__ntrees):
			bootstrap = utils.bootstrap(x.shape[0], self.__random_state)
			self.__trees[i].fit(x[bootstrap, :], y[bootstrap])

	def predict(self, x) -> list:
		if len(self.__trees) == 0:
			raise ValueError("You need to train the model first.")

		y_pred = [self.__trees[i].predict(x) for i in range(len(self.__trees))]
		y_pred = zip(*y_pred)

		return [utils.get_majority_class(list(predictions)) for predictions in y_pred]




