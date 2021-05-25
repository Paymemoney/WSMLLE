import numpy as np

def genMissingLabel(Y, rho):
    J = np.zeros((Y.shape))
    Jpos = np.zeros((Y.shape))
    Jneg = np.zeros((Y.shape))

    temp = np.arange(Y.shape[0])

    for i in range(Y.shape[1]):
        y = Y[:, i]
        pos = temp[y == 1]
        neg = temp[y != 1]

        # pi = np.arange(Y.shape[0])
        pi = np.random.permutation(temp[0:len(pos)])
        pi = pi[0: int(np.round(rho * len(pos)+0.49999999))]
        for k in pos[pi]:
            J[k, i] = 1
            Jpos[k, i] = 1

        # ni = np.arange(Y.shape[0])
        ni = np.random.permutation(temp[0:len(neg)])
        ni = ni[1: int(np.round(rho * len(neg) + 0.49999999))]
        for k in neg[pi]:
            J[k, i] = 1
            Jneg[k, i] = 1

    return J, Jpos, Jneg