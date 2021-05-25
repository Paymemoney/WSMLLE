import numpy as np
from steepest2 import steepest2


def updateD(D, RR, W, para, Jpos, Jneg, Y):
    n = D.shape[0]
    d1 = D.shape[1]

    JposD = Jpos * (D - np.ones((n, d1)))
    JnegD = Jneg * (D + np.ones((n, d1)))
    a = JnegD - JposD
    WDR = W.dot(D.dot(RR))


    x, _ = steepest2(D, W, RR, WDR, a, Y, para)
    return x
