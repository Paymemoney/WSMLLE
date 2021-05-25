import numpy as np

def InitGroup(Y, X, T, Jpos, Jneg, J, para):
    gp = np.unique(T)
    c = para['cluster']
    XPool = {}
    YPool = {}
    JposPool = {}
    JnegPool = {}
    JPool = {}

    for i in range(c):
        k = (T==gp[i]).squeeze()
        XX = X[k,:]
        XPool[i] = XX
        YY = Y[k,:]
        YPool[i] = YY
        JJ = Jpos[k,:]
        JposPool[i] = JJ
        JJ = Jneg[k,:]
        JnegPool[i] = JJ
        JJ = J[k,:]
        JPool[i] = JJ

    return XPool, YPool, JposPool, JnegPool, JPool