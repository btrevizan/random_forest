from pandas import read_csv, concat
from .metrics import Scorer
from math import ceil
from os import path
import numpy as np
import json


datasets = ['credit_g', 'spambase', 'vertebra_column', 'wine']


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


def information_gain(class_entropy, b, index=entropy):
    """Calculate the information gain using an index function.

    :param class_entropy: float
        Target value's entropy.

    :param b: iterable
        Categorical values.

    :param index: list->float (default entropy)
        Index function (entropy or gini).

    :return: float
    """
    return class_entropy - index(b)


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

    :return: iterator
        Each element (a tuple) has the instances' index
        for training set and for the test set, respectively.
    """
    sety = sorted(set(y))
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


def cross_validate(model, x, y, k, random_state):
    """Run a k-fold cross validation.

    :param model: object
        Object with a fit(x, y) and predict([x]) function.

    :param x: matrix
        Instance's attributes.

    :param y: list
        Instance's classes.

    :param k: int
        Number of folds.

    :param random_state: instance of numpy.random.RandomState
        Seed for random generator.

    :return: list
        Test metric for each fold.
    """
    metrics = []

    for train_i, test_i in stratified_split(y, k, random_state):
        x_train = x[train_i, :]
        y_train = y[train_i]

        x_test = x[test_i, :]
        y_test = y[test_i]

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        scorer = Scorer(y_test, y_pred)
        metrics.append(scorer.f1_score())

    return metrics


def load(dataset):
    """Load a dataset.

    :param dataset: str
        Dataset name. Options: credit_g, spambase, vertebra_column, wine

    :return: tuple
        (metadata, x, y, attribute_names), where metadata is a list of numerical features' indexes, x is a matrix and y is a vector, and attribute_names a list of strings.
    """
    default_path = 'tests/datasets/'

    if dataset not in datasets:
        raise ValueError('{} does not exist. The option are credit_g, spambase, vertebra_column, wine.'.format(dataset))

    metadata_path = path.join(default_path, '{}.json'.format(dataset))
    with open(metadata_path, 'r') as file:
        metadata = json.load(file)

    data_path = path.join(default_path, '{}.csv'.format(dataset))
    data = read_csv(data_path, header=0)

    target = None
    numerical_features = []
    attribute_names = []

    for feature in metadata['features']:
        attribute_names.append(feature['name'])
        if feature['type'] == 'numeric':
            #numerical_features.append(feature['name'])
            numerical_features.append( (feature['name'], feature['index']) )  # feature indexes start from 0.
        elif feature['name'] == metadata['default_target_attribute']:
            target = int(feature['index'])

    y = data.iloc[:, target]

    # Encode classes to integers
    sety = set(y.values)
    sety = sorted(sety)
    sety = enumerate(sety)
    sety = list(map(lambda e: (e[1], e[0]), sety))
    sety = dict(sety)

    y = y.apply(lambda e: sety[e], convert_dtype=False).values
    x = concat([data.iloc[:, :target], data.iloc[:, target + 1:]], axis=1, sort=False).values

    return numerical_features, x, y, attribute_names


def majority_voting(y_pred):
    """Group predictions by majority voting.

    :param y_pred: list
        List of predictions.

    :return: int
        Most voted class.
    """
    counts, _ = __unique(y_pred)
    return np.argmax(counts)


def __unique(values):
    _, counts = np.unique(values, return_counts=True)
    return counts, sum(counts)
