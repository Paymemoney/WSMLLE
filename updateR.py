import numpy as np
from steepest import steepest
from scipy.linalg import norm

def updateR(D, R, W, para):
    c = R.shape[0]

    r, _ = steepest(R, D, W)
    for i in range(c):
        R[i:] = r[i:] / norm(r[i:])

    RR = R.dot(R.T)
    return R, RR
