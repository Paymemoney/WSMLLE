import numpy as np
from kernelmatrix import kernelmatrix
from scipy.linalg import inv
from copy import copy

def le_msvr(X, Y, para, vn):
    print('MSVR process with sign consistency and sparsity...\n')

    # proccess the trainDistribution
    N = size(X, 0)
    L = size(Y, 1)

    J = np.zeros((N, L))
    Z = np.zeros((N, L))

    temp = np.arange(Y.shape[1])
    for i in range(N):
        a = Y[i,:]
        v = a[L-1]
        pos = temp[a>=v]
        neg = temp[a<v]
        for k in pos:
            J[i, k] = 1
        for k in neg:
            J[i, k] = -1
        pn = np.sum(J[i,:] == 1)-1
        nn = -(L - pn - 1)
        if pn == 0 | pn == L - 1:
            Z[i, :] = 0
        else:
            for k in pos:
                Z[i, k] = 1. / pn
            for k in neg:
                Z[i, k] = 1. / nn
            Z[i, pos] = 1. / pn
            Z[i, neg] = 1. / nn

    Z[:, -1] = 0

    # build the kernel matrix on the labeled samples (N x N)
    H = kernelmatrix(para['ker'], para['par'], X, X)

    # create matrix for regression parameters
    Beta = np.ones((N, L))
    b = np.ones((1, L))

    # E = prediction error per output (N x L)
    P = H.dot(Beta) + np.tile(b, (N, 1))
    E = Y - P
    E[:, -1] = vn * E[:, -1]
    # compute the Euclidean distance of each examples
    u = np.sqrt(np.sum(E*E, 1))

    # RMSE
    RMSE = []
    RMSE.append(np.sqrt(np.mean(u*u)))

    # points for which prediction error is larger than epsilon, i.e., find SVs whose loss function != 0
    temp = np.arange(u.shape[0])
    i1 = temp[(u >= para['epsi']).squeeze()]

    #　set initial values of alphas (N x 1)
    a = 2 * (u - para['epsi']) / u

    # compute L1. we modify only entries for which  u > epsi
    L1 = np.zeros(u.shape)
    L1[i1] = u[i1]*u[i1] - 2 * para['epsi'] * u[i1] + para['epsi']*para['epsi']
    # L2
    v = P[:, L-1]
    L2 = P - np.tile(v[:, np.newaxis], (1, L))
    L2 = L2 * J
    i2 = (L2 >= 0)
    L2[i2] = 0
    L2 = -np.sum(L2, 1)
    # L3
    L3 = np.sum(Z*P)

    #　Lp is the quantity to minimize (sq norm of parameters + slacks)
    L_T = []
    L_T.append(np.sum(np.diag(Beta.T.dot(H.dot(Beta))))/2 + para['beta1']*np.sum(L1)/2 + para['beta2']*np.sum(L2) - para['beta3']*L3)

    #　initial variables used in loopk
    eta = 1 # step length
    k = 1 # iteration number
    hacer=1 # sentinel of loop
    val = 1 # sign of whether find support vectors

    # strat training
    while (hacer):
        # Print the iteration information
        print('---> Iter:%4d, Omega:%.7f, RMSE:%.7f\n'%(k, L_T[k-1], RMSE[k-1]))

        #　next iteration
        k = k+1

        # save the model parameters in the previous step
        Beta_a = copy(Beta)
        b_a = copy(b)
        i1_a = copy(i1)
        YS_a = copy(J)


        # todo H[i1, :][:, i1]
        M11 = para['beta1'] * H[i1, :][:, i1] + np.eye(size(H[i1, :][:, i1], 0))
        M12 = para['beta1'] * H[i1, :][:, i1].T.dot(a[i1])
        M21 = para['beta1'] * a[i1].T.T.dot(H[i1, :][:, i1])
        M22 = np.zeros((1,1))
        M22[0][0] = para['beta1'] * np.sum(a[i1].T)

        M12 = M12[:, np.newaxis]
        M21 = M21[np.newaxis, :]

        M11 = np.array(M11, dtype=np.float)
        M12 = np.array(M12, dtype=np.float)
        M21 = np.array(M21, dtype=np.float)
        M22 = np.array(M22, dtype=np.float)

        M = np.concatenate((np.concatenate((M11, M12), axis=1), np.concatenate((M21, M22), axis=1)), axis=0)
        M = M + 1e-11 * np.eye(size(M, 0))

        # compute betas
        YS_a[i2] = 0
        M1 = para['beta1'] * Y[i1,:] + (para['beta2'] * YS_a[i1,:] + para['beta3'] * Z[i1,:]) / np.tile(a[i1,np.newaxis], (1, L))
        M2 = para['beta1'] * (a[i1].T.T.dot(Y[i1,:]))+para['beta2']*np.sum(YS_a[i1,:]) + para['beta3']*np.sum(Z)
        sal1=inv(M).dot(np.concatenate((M1, M2[np.newaxis,:]), axis=0))
        b_sal1 = sal1[-1,:]
        b_sal1 = b_sal1[np.newaxis, :]
        sal1 = sal1[:-1,:]

        Beta = np.zeros(size(Beta))
        Beta[i1,:] = sal1
        b = b_sal1

        # recompute error
        P = H.dot(Beta) + np.tile(b, (N, 1))
        E = Y - P
        E[:, -1] = vn * E[:, -1]

        #　recompute i1 and u_z
        u = np.sqrt(np.sum(E*E, 1).astype('float'))
        temp = np.arange(u.shape[0])
        i1 = temp[(u >= para['epsi']).squeeze()]

        # recompute loss function
        L1 = np.zeros(size(u))
        L1[i1] = u[i1] * u[i1] - 2 * para['epsi'] * u[i1] + para['epsi'] * para['epsi']
        v = P[:, L-1]
        L2 = P - np.tile(v[:,np.newaxis], (1, L))
        L2 = L2 * J
        i2 = L2 >= 0
        L2[i2] = 0
        L2 = -np.sum(L2, 1)
        L2 = (L2[:,np.newaxis])[i1,:]
        L3 = np.sum(Z * P)

        L_T.append(np.sum(np.diag(Beta.T.dot(H.dot(Beta)))) / 2 + para['beta1'] * np.sum(L1) / 2 + para['beta2'] * np.sum(L2) - para['beta3'] * L3)
        eta = 1 # initial step
        # Loop where we keep alphas and modify betas
        while (L_T[k-1] > L_T[k - 2]):
            eta = eta / 10 # modify step length
            i1 = i1_a #　restore i1

            Beta = np.zeros(size(Beta))
            Beta[i1,:]=eta * sal1 + (1 - eta) * Beta_a[i1,:]
            b = eta * b_sal1 + (1 - eta) * b_a

            #　recoumpte
            P = H.dot(Beta) + np.tile(b, (N, 1))
            E = Y - P
            E[:, -1] = vn * E[:, -1]
            u = np.sqrt(np.sum(E*E, 1).astype('float'))
            temp = np.arange(u.shape[0])
            i1 = temp[(u >= para['epsi']).squeeze()]

            L1 = np.zeros(size(u))
            L1[i1] = u[i1]*u[i1] - 2 * para['epsi'] * u[i1] + para['epsi'] * para['epsi']
            v = P[:, L-1]
            L2 = P - np.tile(v[:, np.newaxis], (1, L))
            L2 = L2 * J
            i2 = L2 >= 0
            L2[i2] = 0
            L2 = -np.sum(L2, 1)
            L2 = L2[:, np.newaxis][i1, :]
            L3 = np.sum(Z * P)

            L_T[k-1] = np.sum(np.diag(Beta.T.dot(H.dot(Beta)))) / 2 + para['beta1'] * np.sum(L1) / 2 + para['beta2'] * np.sum(L2) - para['beta3'] * L3

            #　stopping criterion #1
            if (eta < 1e-16):
                L_T[k-1] = L_T[k - 2] - 1e-15
                # save parameters
                Beta = copy(Beta_a)
                b = copy(b_a)
                i1 = copy(i1_a)
                hacer = 0 # stop loop

        a_a = a
        a = 2 * (u - para['epsi']) / u
        RMSE.append(np.sqrt(np.mean(u*u)))

        # stopping criterion #2
        if ((L_T[k - 2]- L_T[k-1]) / L_T[k - 2] < para['tol']):
            print('---> Iter:%4d, Omega:%.7f, RMSE:%.7f\n'%(k, L_T[k-1], RMSE[k-1]))
            hacer = 0 # stop loop

        # stopping criterion #3 - algorithm does not converge. (val = -1)
        if (len(i1)==0):
            print('---> Stop: algorithm does not converge (find no SVs).\n')
            Beta = np.zeros(size(Beta))
            b = np.zeros(size(b))
            i1 = []
            val = -1
            hacer = 0 # stop loop

    # save model
    model = {}
    model['Beta'] = Beta
    model['b'] = b
    model['svindex'] = i1
    model['ker'] = para['ker']
    model['par'] = para['par']

    return model



def size(X, i=None):
    if(i!=None):
        return X.shape[i]
    else:
        return X.shape