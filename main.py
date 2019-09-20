import numpy
import src.random_forest as rf
import src.dtree as dtree
import src.utils as utils
import graphviz

seed = 0
ntrees = 30
dataset = 'credit_g'

#

data = utils.load(dataset)
x = data[1]  # table
y = data[2]  # classes

data[3].remove('class')
dtree.all_attribute_names = data[3]

#

random_state = numpy.random.RandomState(seed)
random_forest = rf.RandomForest(ntrees, random_state, x, y, data[0])

#

for i in range(5):
	dot = graphviz.Digraph(name=str(i))
	random_forest.trees[i].get_graph(dot)
	dot.render(cleanup=True)

#

