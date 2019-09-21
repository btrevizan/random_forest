import numpy as np


class Scorer:
    """Encapsulate all metrics.

    :param y_true: list
        True labels.

    :param y_pred: list
        Predicted labels.
    """

    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        self.n = len(y_true)
        self.cm = None

        self.__make()

    def tp(self, label):
        """True positive (TP) of class 'label'."""
        return self.cm[label, label]

    def tn(self, label):
        """True negative (TN) of class 'label'."""
        return sum([self.tp(i) for i in range(self.cm.shape[0]) if i != label])

    def fp(self, label):
        """False positive (FP) of class 'label'."""
        return sum([self.cm[i, label] for i in range(self.cm.shape[0]) if i != label])

    def fn(self, label):
        """False negative (FN) of class 'label'."""
        return sum([self.cm[label, i] for i in range(self.cm.shape[1]) if i != label])

    def accuracy(self, label=1):
        """Calculate accuracy.

        :param label: int (optional)
            Class label.

        :return float
        """
        return (self.tp(label) + self.tn(label)) / self.n

    def recall(self, label=1):
        """Calculate recall/sensitivity/TPR.

        :param label: int (optional)
            Class label.

        :return float
        """
        tp = self.tp(label)
        return tp / (tp + self.fn(label))

    def precision(self, label=1):
        """Calculate precision.

        :param label: int (optional)
            Class label.

        :return float
        """
        tp = self.tp(label)
        return tp / (tp + self.fp(label))

    def f1_score(self, label=0):
        """Calculate F1-Score for binary classification and micro-average F1-Score for multi-class.

        :param label: int (optional, ignored)
            Class label.

        :return float
        """
        n, _ = self.cm.shape

        if n == 2:
            precision = self.precision()
            recall = self.recall()
        else:
            tp = np.sum([self.tp(i) for i in range(n)])
            precision = tp / (tp + np.sum([self.fp(i) for i in range(n)]))
            recall = tp / (tp + np.sum([self.fn(i) for i in range(n)]))

        return (2 * precision * recall) / (precision + recall)

    def balanced_accuracy(self):
        """Calculate balance accuracy.

        :return: float
        """
        n, _ = self.cm.shape
        recall = [self.recall(i) for i in range(n)]
        return np.sum(recall) / n

    def __make(self):
        sety_true, counts_true = np.unique(self.y_true, return_counts=True)
        n_labels = len(sety_true)

        self.cm = np.zeros((n_labels, n_labels))
        for label_true, count_true in zip(sety_true, counts_true):

            true_i = set(np.where(self.y_true == label_true)[0])

            for label_pred in sety_true:
                prediction_i = set(np.where(self.y_pred == label_pred)[0])
                self.cm[label_true, label_pred] = len(true_i.intersection(prediction_i))

        return self.cm
