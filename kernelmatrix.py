import numpy as np

def kernelmatrix(ker, parameter, testX, trainX):
    ok = 0
    if(ker=='lin'):
        ok = 1
        if(type(trainX)!=np.ndarray):
            K = testX.dot(trainX.T) + parameter
        else:
            K = testX.dot(testX.T) + parameter

    if (ker == 'poly'):
        ok = 1
        if (type(trainX)!=np.ndarray):
            K = testX.dot(trainX.T) + parameter
        else:
            K = testX.dot(testX.T) + parameter

    if (ker == 'rbf'):
        ok = 1
        n1sq = np.sum(testX.T * testX.T, 0) # compute x^2
        n1 = testX.T.shape[1]
        if (type(trainX)!=np.ndarray):
            # ||x-y||^2 = x^2 + y^2 - 2*x'*y
            D = (np.ones((n1, 1)).dot(n1sq)).T + np.ones((n1,1)).dot(n1sq) -2*testX.dot(testX.T)
        else:
            n2sq = np.sum(trainX.T*trainX.T, 0)
            n2 = trainX.T.shape[1]
            #　||x-y||^2 = x^2 + y^2 - 2*x'*y
            D = (np.ones((n2, 1)).dot(n1sq[np.newaxis,:])).T + np.ones((n1,1)).dot(n2sq[np.newaxis,:]) -2*testX.dot(trainX.T)
        K = np.exp(-D / (2 * parameter * parameter))

    if (ker == 'sam'):
        ok = 1
        if (type(trainX)!=np.ndarray):
            D = testX.dot(trainX.T)
        else:
            d = testX.dot(testX.T)
        K = np.exp(-np.arccos(D)*np.arccos(D) / (2 * parameter ^ 2))

    if(ok == 0):
        print('Unsupported kernel', ker)
        exit()

    return K