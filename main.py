import numpy
import src.random_forest as rf
import src.utils as utils
import graphviz

seed = 0
ntrees = 30
dataset = 'wine'

#

data = utils.load(dataset)
x = data[1]  # table
y = data[2]  # classes

import code
code.interact(local=locals())

#

random_state = numpy.random.RandomState(seed)
random_forest = rf.RandomForest(ntrees, random_state, x, y, data[0])

#

for i in range(5):
	dot = graphviz.Digraph(name=str(i))
	random_forest.trees[i].get_graph(dot)
	dot.render(cleanup=True)

#

