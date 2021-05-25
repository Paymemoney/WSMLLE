import numpy as np
from sklearn.metrics import roc_auc_score


def avgauc( pred, target):
    avgauc = 0
    for i in range(pred.shape[1]):
        y = pred[:, i]
        t = target[:, i]
        if (np.mean(t) == 1 or np.mean(t) == -1):
            if (np.mean(t) == 1):
                auc_tmp = 1
            else:
                auc_tmp = 0
        else:
            auc_tmp = roc_auc_score(t, y)
        avgauc = avgauc + auc_tmp / pred.shape[1]

    return avgauc