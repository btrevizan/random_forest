from src.random_forest import RandomForest
import src.utils as utils
import numpy as np
import graphviz

seed = 0
ntrees = 30
dataset = 'credit_g'

#

numerical_attributes, attr_names, x, y = utils.load(dataset)

#

random_state = np.random.RandomState(seed)

random_forest = RandomForest(ntrees, random_state, numerical_attributes)
random_forest.fit(x, y)

#

for i in range(5):
	dot = graphviz.Digraph(name=str(i))
	random_forest.trees[i].get_graph(dot, attr_names=attr_names)
	dot.render(cleanup=True)

#

