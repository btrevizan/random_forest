"""Feature selection for spambase dataset."""
from src.utils import information_gain, entropy
from pandas import read_csv, concat, DataFrame

data = read_csv('tests/datasets/spambase.csv', header=0, index_col=None)
x, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
n, d = x.shape

gains = []
class_entropy = entropy(y)
for j in range(d):
    info_gain = information_gain(class_entropy, x[:, j], y)
    gains.append((j, info_gain))

gains.sort(key=lambda t: t[1], reverse=True)
for gain in gains:
    print(gain)

js, _ = zip(*gains)
js = list(js)

data = concat([data.iloc[:, js[:10]], data.iloc[:, -1]], axis=1)
data.to_csv('tests/datasets/spambase_processed.csv', header=True, index=False)
