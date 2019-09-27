from src.dtree import DecisionTree
from src.utils import load
import graphviz

numerical_attributes, attribute_names, attribute_values, x, y = load('benchmark')

dtree = DecisionTree(None, numerical_attributes, attribute_values, benchmark=True)
dtree.fit(x, y)

dot = graphviz.Digraph(name='tests/results/benchmark_tree')
dtree.get_graph(dot, attr_names=attribute_names)
dot.render(cleanup=True)
