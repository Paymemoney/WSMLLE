import numpy as np
from scipy import sparse
from scipy.sparse.linalg import inv

def ADDM(A, b, lmbda, rho, alpha):
    global ITER,AB1,AB2
    ITER = 10
    AB1 = 1e-4
    AB2 = 1e-2
    (m, n) = A.shape
    Atb = A.T.dot(b)
    x = np.zeros((n, 1))
    z = np.zeros((n, 1))
    u = np.zeros((n, 1))

    L, U = factor(A, rho)

    for k in range(ITER):
        q = Atb[:,np.newaxis] + rho * (z - u)
        if (m >= n):
            x = inv(U).dot(inv(L).dot(q))
        else:
            temp = A.T.dot(inv(U).dot(inv(L).dot(A.dot(q))))
            x = q / rho - temp / (rho*rho)

        zold = z
        x_hat = alpha * x + (1 - alpha) * zold
        z = shrinkage(x_hat + u, lmbda /rho)

        u = u + (x_hat - z)

        if(np.linalg.norm(x-z, 2) < np.sqrt(n)*AB1 + AB2*max(np.linalg.norm(x, 2), np.linalg.norm(-z, 2))\
                and np.linalg.norm(-rho*(z-zold), 2) < np.sqrt(n)*AB1 + AB2*np.linalg.norm(rho*u, 2)):
            break

        # if np.linalg.norm(u, 2)<1e-4:
        #     print('Converged after %i iterations.' % k)
        #     break
    return z


def factor(A, rho):
    (m, n) = A.shape
    if (m >= n):
        # L = chol(A.T*A + rho*np.eye(n), 'lower' )
        L = np.linalg.cholesky(A.T*A + rho*np.eye(n))
    else:
        # L = chol(np.eye(m) + 1 / rho * (A * A.T), 'lower' )
        L = np.linalg.cholesky(np.eye(m) + 1 / rho * (A.dot(A.T)))
    L = sparse.coo_matrix(L).tocsr()
    U = sparse.coo_matrix(L.T).tocsr()
    # U = L.T
    return L,U

def objective(A, b, lmbda, x, z):
    return ( 1/2*np.sum((A*x - b)*(A*x - b)) + lmbda*np.norm(z,1))

def shrinkage(x, kappa):
    return np.maximum(0, x-kappa) - np.maximum(0, -x-kappa)