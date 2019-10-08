"""Feature selection for spambase dataset."""
from src.utils import information_gain, entropy
from pandas import read_csv, concat
import numpy as np
import math


def numerical_information_gain(target_entropy, data_x, data_y):
    discretized_x = None
    max_gain = -math.inf

    new_y = data_y[np.argsort(data_x)]
    thresholds = [(data_x[i - 1] + data_x[i]) / 2 for i in range(1, len(new_y)) if new_y[i - 1] != new_y[i]]

    for t in thresholds:
        new_x = np.zeros(data_x.shape)
        new_x[data_x > t] = 1

        ig = information_gain(target_entropy, new_x, data_y)
        if ig > max_gain:
            max_gain = ig
            discretized_x = new_x

    return information_gain(target_entropy, discretized_x, data_y)


data = read_csv('tests/datasets/spambase.csv', header=0, index_col=None)
x, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
n, d = x.shape

gains = []
class_entropy = entropy(y)
for j in range(d):
    info_gain = numerical_information_gain(class_entropy, x[:, j], y)
    gains.append((j, info_gain))

gains.sort(key=lambda t: t[1], reverse=True)
for gain in gains:
    print(gain)

js, _ = zip(*gains)
js = list(js)

data = concat([data.iloc[:, js[:6]], data.iloc[:, -1]], axis=1)
data.to_csv('tests/datasets/spambase_processed.csv', header=True, index=False)
