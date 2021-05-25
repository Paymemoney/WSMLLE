import numpy as np
import ADMM

def estimate_W(X):
    print('Structural information discovery...\n')
    N = X.shape[0]
    W = np.zeros((N,N))

    for i in range(N):
        A = X
        A = np.delete(A, 0, axis=0)
        b = X[i,:].T
        lmbda_max = np.linalg.norm(A.dot(b), np.inf)
        lmbda = 0.01*lmbda_max
        alpha = 0.5
        rho = 1
        z = ADMM.ADDM(A.T, b, lmbda, rho, alpha)
        z = np.insert(z,i,[0])
        W[i,:] = z

    return W