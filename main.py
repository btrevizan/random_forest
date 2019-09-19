import numpy
import src.random_forest as rf
import src.utils as utils
import graphviz

seed = 0
ntrees = 7
dataset = 'credit_g'

#

data = utils.load(dataset)
x = data[1]  # table
y = data[2]  # classes

#

random_state = numpy.random.RandomState(seed)
random_forest = rf.RandomForest(ntrees, random_state, x, y, data[0])

#

dot = graphviz.Digraph()
random_forest.trees[0].get_graph(dot)

dot.render(cleanup=True)
