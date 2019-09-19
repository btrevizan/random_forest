import numpy
import src.random_forest as rf
import src.utils as utils

seed = 0
ntrees = 10
dataset = 'credit_g'

#

data = utils.load(dataset)
x = data[1]  # table
y = data[2]  # classes

#

random_state = numpy.random.RandomState(seed)
random_forest = rf.RandomForest(ntrees, random_state, x, y)



