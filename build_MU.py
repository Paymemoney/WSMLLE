import numpy as np
from cvxopt import solvers, matrix

def build_MU(Y,W,C,J):
    print('Reconstruct the label space...\n')
    N = Y.shape[0]
    L = Y.shape[1]
    M = np.eye(N,N)
    for i in range(N):
        w = W[i,:]
        M[i,:] = M[i,:] - w
        M[:,i] = M[:,i] - w.T
        M = M + w.T.dot(w)

    M[np.isnan(M)] = 0
    M[np.isinf(M)] = 0
    M = matrix(M)
    # QP
    b = matrix(np.zeros((N,1)) - C['c1'])
    # options = optimoptions('quadprog','Display', 'off')
    YY = Y * J # 按元素乘
    YY[YY == 0] = -1
    MU = matrix(np.zeros((N, L)))
    temp = matrix(np.zeros(N))
    for k in range(L):
        A = matrix(-np.diag(YY[:,k]))
        # lb = matrix(-C['c2'] * np.ones((N,1))) #lower bound
        # ub = matrix(C['c2'] * np.ones((N,1))) #upper bound
        # solvers.options['show_progress'] = False
        # 下面解二次规划参数可能有问题
        MU[:,k] = solvers.qp(2*M, temp, G = None, h = None, A = A, b = b)['x']

    MU = np.array(MU)
    return MU