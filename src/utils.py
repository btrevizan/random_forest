import numpy as np
from math import floor, ceil


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
        result = result - p * np.log2(p)

    return result


def gini(values):
    """Get the values' Gini index.

    :param values: iterable
        Categorical values.

    :return: float
    """
    counts, total = __unique(values)
    result = [pow(n / total, 2) for n in counts]
    return 1 - np.sum(result)


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


def bootstrap(n, random_state):
    """Create a list of random integers between 0 and n-1 with length n.

    :param n: int
        Number of instances.

    :param random_state: instance of numpy.random.RandomState
        Seed for random generator.

    :return: list
        List with numbers between 0 and n-1 (instances' index)
    """
    return random_state.randint(low=0, high=n, size=n)


def stratified_split(y, n_splits, random_state):
    """Create k folds for k-fold cross validation.

    :param y: list
        Instances' label.

    :param n_splits: int
        Number of splits (folds).

    :param random_state: instance of numpy.random.RandomState
        Seed for random generator.

    :return: list of tuples
        Each element of the list (a tuple) has the instances' index
        for training set and for the test set, respectively.
    """
    sety = set(y)
    classes_indexes = []

    for c in sety:
        indexes = list(random_state.permutation(np.where(y == c)[0]))
        length = ceil(len(indexes) / n_splits)
        classes_indexes.append((indexes, length))

    folds = []
    for i in range(n_splits):
        folds.append([])
        for indexes, n in classes_indexes:
            folds[i] += indexes[i * n:(i * n) + n]

    for i in range(n_splits):
        test = folds[i]
        train = [index for indexes in folds[0:i] + folds[i+1:] for index in indexes]

        yield train, test


def __unique(values):
    _, counts = np.unique(values, return_counts=True)
    return counts, sum(counts)
