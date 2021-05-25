import numpy as np
from kernelmatrix import kernelmatrix
from Ranking_loss import Ranking_loss
from One_error import One_error
from coverage import coverage
from Average_precision import Average_precision
from MacroF1 import MacroF1
from MicroF1 import MicroF1
from avgauc import avgauc

def mytest(trainX, testX, testY, model):
    Ktest = kernelmatrix(model['ker'], model['par'], testX, trainX)
    degree = Ktest.dot(model['Beta']) + np.tile(model['b'], (Ktest.shape[0], 1))
    label = np.zeros(degree.shape)
    vd = degree[:, -1]

    # todo 这种索引可以吗
    for i in range(degree.shape[0]):
        label[i, degree[i,:] >= vd[i]] = 1
        label[i, degree[i,:] < vd[i]] = -1

    degree = np.delete(degree, -1, axis=1)
    label = np.delete(label, -1, axis=1)
    Pre_Labels = label.T

    Max = np.max(degree)
    Min = np.min(degree)
    Outputs = (degree - Min) / (Max - Min)
    Outputs = Outputs.T

    RankingLoss = Ranking_loss(Outputs, testY)
    OneError = One_error(Outputs, testY)
    Coverage = coverage(Outputs, testY)
    Average_Precision = Average_precision(Outputs, testY)
    MacF1 = MacroF1(Pre_Labels, testY)
    MicF1 = MicroF1(Pre_Labels, testY)
    AvgAuc = avgauc(Pre_Labels, testY)
    return OneError, Coverage, RankingLoss, Average_Precision, AvgAuc, MacF1, MicF1