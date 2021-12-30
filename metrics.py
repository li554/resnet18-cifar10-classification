import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


class Metric(object):
    def __init__(self, output, label):
        self.output = output
        self.label = label

    def accuracy(self):
        y_pred = self.output
        y_true = self.label
        y_pred = y_pred.argmax(dim=1)
        accuracy = accuracy_score(y_true, y_pred)
        return accuracy

    def f1_score(self, _type='micro'):
        y_pred = self.output
        y_true = self.label
        return f1_score(np.argmax(y_pred, 1), y_true, average=_type)
