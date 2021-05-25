import random
import numpy as np
import matplotlib.pyplot as plt

"""
最速下降法
Rosenbrock函数
函数 f(x) = 2*x(1)^2+x(2)^2
梯度 g(x)=(4*x(1)),2*x(2))^(T)
"""


def goldsteinsearch(f, df, d, x, alpham, rho, t):
    '''
    线性搜索子函数
    数f，导数df，当前迭代点x和当前搜索方向d
    '''
    flag = 0

    a = 0
    b = alpham
    fk = f(x)
    gk = df(x)

    phi0 = fk
    dphi0 = np.dot(gk, d)
    # print(dphi0)
    alpha = b * random.uniform(0, 1)

    while (flag == 0):
        newfk = f(x + alpha * d)
        phi = newfk
        # print(phi,phi0,rho,alpha ,dphi0)
        if (phi - phi0) <= (rho * alpha * dphi0):
            if (phi - phi0) >= ((1 - rho) * alpha * dphi0):
                flag = 1
            else:
                a = alpha
                b = b
                if (b < alpham):
                    alpha = (a + b) / 2
                else:
                    alpha = t * alpha
        else:
            a = a
            b = alpha
            alpha = (a + b) / 2
    return alpha


def rosenbrock(x):
    # 函数:f(x) = 2*x(1)^2+x(2)^2
    return 2 * x[0] ** 2 + x[1] ** 2


def jacobian(x):
    # 梯度 g(x)=(4*x(1)),2*x(2))^(T)
    return np.array([4 * x[0], 2 * x[1]])


def steepest(x0):
    print('初始点为:')
    print(x0, '\n')
    imax = 20000
    W = np.zeros((2, imax))
    epo = np.zeros((2, imax))
    W[:, 0] = x0
    i = 1
    x = x0
    grad = jacobian(x)
    delta = sum(grad ** 2)  # 初始误差

    f = open("最速.txt", 'w')

    while i < imax and delta > 10 ** (-5):
        p = -jacobian(x)
        x0 = x
        alpha = goldsteinsearch(rosenbrock, jacobian, p, x, 1, 0.1, 2)
        x = x + alpha * p
        W[:, i] = x

        epo[:, i] = np.array((i, delta))
        f.write(str(i) + "        " + str(delta) + "\n")  #
        print(i, np.array((i, delta)))
        grad = jacobian(x)
        delta = sum(grad ** 2)
        i = i + 1
    print("迭代次数为:", i)
    print("近似最优解为:")
    print(x, '\n')
    W = W[:, 0:i]  # 记录迭代点
    return [W, epo]


if __name__ == "__main__":
    X1 = np.arange(-1.5, 1.5 + 0.05, 0.05)
    X2 = np.arange(-3.5, 4 + 0.05, 0.05)
    [x1, x2] = np.meshgrid(X1, X2)

    f = 2 * x1 ** 2 + x2 ** 2
    plt.contour(x1, x2, f, 20)  # 画出函数的20条轮廓线
    x0 = np.array([1, 1])
    list_out = steepest(x0)
    W = list_out[0]
    epo = list_out[1]
    plt.plot(W[0, :], W[1, :], 'g*-')  # 画出迭代点收敛的轨迹
    plt.show()