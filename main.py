import numpy
import src.random_forest as rf
import src.utils as utils

seed = 0
ntrees = 1
dataset = 'wine'

#

data = utils.load(dataset)
table = data[1]
classes = data[2]

#

# random_state = numpy.random.RandomState(seed)
# random_forest = rf.RandomForest(ntrees, random_state, table, classes)



