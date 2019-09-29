"""Feature selection for spambase dataset."""
from sklearn.feature_selection import RFE  # Recursive Feature Elimination (ranker)
from pandas import read_csv, concat
from sklearn.svm import SVC
import numpy as np

data = read_csv('tests/datasets/spambase.csv', header=[0], index_col=None)
x, y = data.iloc[:, :-1], data.iloc[:, -1]

estimator = SVC(kernel='linear', gamma='auto', random_state=np.random.RandomState(22))
selector = RFE(estimator)
selector = selector.fit(x, y)

selected_mask = selector.support_

x = x.iloc[:, selected_mask]

data = concat([x, y], axis=1)
data.to_csv('tests/datasets/spambase_processed.csv', header=True, index=False)

