import numpy as np
from sklearn import metrics

def evaluate(predict, label):
    aupr = metrics.average_precision_score(y_true=label, y_score=predict)
    auroc = metrics.roc_auc_score(y_true=label, y_score=predict)
    result = {"aupr": aupr,
              "auroc": auroc}
    return result
