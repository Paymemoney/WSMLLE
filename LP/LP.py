import numpy as np

def LPA(X,label,C,dd):
    # label有标号数据点所属类
    # l有标号的数据点个数
    # N=l+u,数据点个数
    # C类别个数
    # dd迭代次数
    D = X.shape[0]
    N = X.shape[1]
    l = label.size

    # 计算距离
    X2 = np.sum(X*X, 0)
    distance = np.tile(X2, (N, 1)) + np.tile(X2.T,(1,N))-2*X.T.dot(X)

    # Step2: 构造相似矩阵W
    W = np.zeros(N, N)
    t = 2
    for i in range(N):
        for j in range(N):
            W[i,j] = np.exp(-distance(i,j)/(t*t))

    # Step3: 构造概率传递矩阵T（P)
    T = np.zeros((N,N))

    zz = 1 / np.sum(W, 1)
    for i in range(N):
        T[i,:]=W[i,:]*zz[i]

    Y = np.ones((N, C)) * 0.5

    # 归一化
    for i in range(N):
        for j in range(C):
            Y[i,j] = Y[i,j]/np.sum(Y[i,:])

    for i in range(l):
        j = label[i]
        Y[i, :] = 0
        Y[i, j] = 1

    # 更新自己的概率分布
    F = Y
    for i in range(dd):
        F = T.dot(F)
        for i in range(l):
            j = label[i]
            F[i, :] = 0
            F[i, j] = 1

    # 找到每个数据所属于的类
    B = np.sort(F,0)
    index = np.argsort(F,0)
    B = B[::-1]
    index = index[::-1]

    Labelnew = index[1,:]
    return Labelnew
