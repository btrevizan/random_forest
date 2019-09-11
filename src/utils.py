import numpy as np
from math import log2, fsum


def entropy(values):
    """Get the values' entropy (measure of randomness).

    :param values: iterable
        Categorical values.

    :return: float
    """
    counts, total = __unique(values)

    result = 0
    for n in counts:
        p = n / total
        result = result - p * log2(p)

    return result


def gini(values):
    """Get the values' Gini index.

    :param values: iterable
        Categorical values.

    :return: float
    """
    counts, total = __unique(values)
    result = [pow(n / total, 2) for n in counts]
    return 1 - fsum(result)


def information_gain(a, b, index=entropy):
    """Calculate the information gain using an index function.

    :param a: iterable
        Categorical values.

    :param b: iterable
        Categorical values.

    :param index: list->float (default entropy)
        Index function (entropy or gini).

    :return: float
    """
    return index(a) - index(b)


def bootstrap(n, seed):
    """Create a list of random integers between 0 and n-1 with length n.

    :param n: int
        Number of instances.

    :param seed: int
        Seed for random generator.

    :return: list
        List with numbers between 0 and n-1 (instances' index)
    """
    rs = np.random.RandomState(seed)
    return rs.randint(low=0, high=n, size=n)


def __unique(values):
    _, counts = np.unique(values, return_counts=True)
    return counts, sum(counts)
