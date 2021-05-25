import numpy as np
from scipy.cluster.vq import kmeans
from InitGroup import InitGroup
from estimate_W import estimate_W
from build_MU import build_MU
from updateR import updateR
from updateD import updateD
import json

def LE(iii, J, Jpos, Jneg, X, Y, para):
    c = Y.shape[1]

    # kmeans
    # T,_ = kmeans(X, para['cluster'])
    T = np.ones((297,1))
    XPool, YPool, JposPool, JnegPool, JPool = InitGroup(Y, X, T, Jpos, Jneg, J, para)
    XP = cell2mat(XPool, para['cluster'])
    YP = cell2mat(YPool, para['cluster'])
    J = cell2mat(JPool, para['cluster'])
    Jpos = cell2mat(JposPool, para['cluster'])
    Jneg = cell2mat(JnegPool, para['cluster'])


    # for i in range(para['cluster']):
    #     if i==0:
    #         XP = XPool[0]
    #         YP = YPool[0]
    #         J = JPool[0]
    #         Jpos = JposPool[0]
    #         Jneg = JnegPool[0]
    #     else:
    #         XP = np.concatenate(XP, XPool[i])
    #         YP = np.concatenate(YP, YPool[i])
    #         J = np.concatenate(J, JPool[i])
    #         Jpos = np.concatenate(Jpos, JposPool[i])
    #         Jneg = np.concatenate(Jneg, JnegPool[i])

    # manifold for W
    W = {}
    for i in range(para['cluster']):
        W[i] = estimate_W(XPool[i])

    # initialize for R
    RPool = {}
    for i in range(para['cluster']):
        RPool[i] = np.random.rand(c,c)
    RRPool = {}
    for i in range(para['cluster']):
        RRPool[i] = RPool[i].dot(RPool[i].T)

    # initialize for D
    DPool = {}
    C = {}
    C['c1'] = 1
    C['c2'] = 2
    for i in range(para['cluster']):
        DPool[i] = build_MU(YPool[i], W[i], C, JPool[i])

    T = 'kmeans_' + str(para['cluster']) + '_' + str(iii) + '.json'
    Init = {}
    Init['XPool'] = str(XPool)
    Init['YPool'] = str(YPool)
    Init['JposPool'] = str(JposPool)
    Init['JnegPool'] = str(JnegPool)
    Init['JPool'] = str(JPool)
    Init['W'] = str(W)
    Init['RPool'] = str(RPool)
    Init['RRPool'] = str(RRPool)
    Init['DPool'] = str(DPool)
    js_save(T, Init)


    ###
    # obj function
    loss = lossfun(DPool, RRPool, W, Jpos, Jneg, para)
    loss = [loss]
    for i in range(para['maxIter']):
        for j in range(para['cluster']):
            r, RR = updateR(DPool[j], RPool[j], W[j], para)
            RPool[j] = r
            RRPool[j] = RR
            d = updateD(DPool[j], RRPool[j], W[j], para, JposPool[j], JnegPool[j], YPool[j])
            DPool[j] = d

        Yenhacement = cell2mat(DPool, para['cluster'])

        loss_now = lossfun(DPool, RRPool, W, Jpos, Jneg, para)
        loss.append(loss_now)

        if i<5:
            continue
        for ii in range(3):
            stopnow = (np.abs(loss_now - loss[i - ii]) < 1e-6)
        if stopnow:
            break
    return XP, YP, Yenhacement


def cell2mat(XPool, cluster):
    for i in range(cluster):
        if i==0:
            XP = XPool[0]
        else:
            XP = np.concatenate(XP, XPool[i])
    return XP


def js_save(filename, file):
    with open(filename, 'w') as file_obj:
        json.dump(file, file_obj)

def lossfun(DPool, RRPool, W, Jpos, Jneg, para):
    D = cell2mat(DPool, para['cluster'])
    n = D.shape[0]
    d1 = D.shape[1]
    JposD = Jpos*(D - np.ones((n, d1)))
    JnegD = Jneg*(D + np.ones((n, d1)))
    ll = np.sum(np.tanh(JnegD - JposD))
    for i in range(para['cluster']):
        k = DPool[i] - W[i].dot(DPool[i].dot(RRPool[i]))
        l = ll + para['lamda'] * np.trace(k.dot(k.T))
    return l