import numpy as np
import sys
import os
from scipy.io import loadmat, savemat
from scipy.spatial.distance import pdist, squareform
from genMissingLabel import genMissingLabel
from LE import LE
from le_msvr import le_msvr
from mytest import mytest
from softmax import softmax

# sys.path.append()

OneError_ten = []
Coverage_ten = []
RankingLoss_ten = []
AvePre_ten = []
AvgAuc_ten = []
MacF1_ten = []
MicF1_ten = []

observePercentage = 0.8
virtualNum = 5

para = {}
para['lamda'] = 1e-4
para['tooloptions'] = {}
para['tooloptions']['maxiter'] = 10
para['tooloptions']['gradnorm'] = 1e-3
para['cluster'] = 1
para['maxIter'] = 10

parasvr = {}
parasvr['tol']  = 1e-3
parasvr['epsi'] = 0
parasvr['beta1'] = 10
parasvr['beta2'] = 1
parasvr['beta3'] = 8
parasvr['ker'] = 'rbf'


for i in range(1,11):
    # T = 'DataSets_new2\emotions_total_' + str(i) + '.mat'
    T = 'emotions\emotions_total_' + str(i) + '.mat'

    file = loadmat(T)

    Xtrain = file['train_data']
    Ytrain = file['train_target'].T
    Xtest = file['test_data']
    Ytest = file['test_target'].T


    parasvr['par'] = 1 * np.mean(pdist(Xtrain))
    n = Ytrain.shape[0]
    c = Ytrain.shape[1]

    J, Jpos, Jneg = genMissingLabel(Ytrain, observePercentage)
    XP, YP, Yenhacement = LE(i, J, Jpos, Jneg, Xtrain, Ytrain, para)
    T = 'XY_' + str(i) + '.mat'
    mat = {}
    mat['XP'] = XP
    mat['YP'] = YP
    mat['Yenhacement'] = Yenhacement
    savemat(T, mat)

    v = np.tile(np.zeros((Yenhacement.shape[0], 1)), (1, virtualNum))
    Yenhacementv = np.concatenate((Yenhacement, v), axis=1)
    Yenhacementv = (softmax(Yenhacementv.T)).T # todo

    if virtualNum > 1:
        Yenhacementv = Yenhacementv[:, : c+1]

    parasvr['tol'] = 1e-3
    parasvr['epsi'] = 0
    parasvr['beta1'] = 10
    parasvr['beta2'] = 1
    parasvr['beta3'] = 8
    parasvr['ker'] = 'rbf'

    model = le_msvr(XP, Yenhacementv, parasvr, virtualNum)
    T = 'model_'+str(i)+'.mat'
    savemat(T, model)

    o, c, r, ap, aa, mac, mic = mytest(XP, Xtest, Ytest.T, model)

    OneError_ten.append(o)
    Coverage_ten.append(c)
    RankingLoss_ten.append(r)
    AvePre_ten.append(ap)
    AvgAuc_ten.append(aa)
    MacF1_ten.append(mac)
    MicF1_ten.append(mic)

mat = {}
mat['OneError_ten'] = OneError_ten
mat['Coverage_ten'] = Coverage_ten
mat['RankingLoss_ten'] = RankingLoss_ten
mat['AvePre_ten'] = AvePre_ten
mat['AvgAuc_ten'] = AvgAuc_ten
mat['MacF1_ten'] = MacF1_ten
mat['MicF1_ten'] = MicF1_ten
savemat('result_ten.mat', mat)
result = {}
result['alpha_1'] = np.std(OneError_ten)
result['alpha_2'] = np.std(Coverage_ten)
result['alpha_3'] = np.std(RankingLoss_ten)
result['alpha_4'] = np.std(AvePre_ten)
result['alpha_5'] = np.std(AvgAuc_ten)
result['alpha_6'] = np.std(MacF1_ten)
result['alpha_7'] = np.std(MicF1_ten)

result['OneError'] = np.mean(OneError_ten)
result['Coverage'] = np.mean(Coverage_ten)
result['RankingLoss'] = np.mean(RankingLoss_ten)
result['Average'] = np.mean(AvePre_ten)
result['AvgAuc'] = np.mean(AvgAuc_ten)
result['MacF1'] = np.mean(MacF1_ten)
result['MicF1'] = np.mean(MicF1_ten)
savemat('result.mat', result)

