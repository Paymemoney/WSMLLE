import random
import numpy as np
import matplotlib.pyplot as plt

"""
最速下降法
Rosenbrock函数
函数 f(x) = 2*x(1)^2+x(2)^2
梯度 g(x)=(4*x(1)),2*x(2))^(T)
"""


def goldsteinsearch(f, df, d, x, alpham, rho, t, W, RR, WDR, a, Y, para):
    '''
    线性搜索子函数
    数f，导数df，当前迭代点x和当前搜索方向d
    '''
    flag = 0

    a = 0
    b = alpham

    fk = f(x, W, RR, a, para)
    gk = df(x, WDR, W, RR, a, para, Y)

    phi0 = fk
    dphi0 = np.sum(gk * d)
    # print(dphi0)
    alpha = b * random.uniform(0, 1)

    while (flag == 0):
        newfk = f(x + alpha * d, W, RR, a, para)
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


def rosenbrock(x, W, RR, a, para):
    kk = x - W.dot(x.dot(RR))
    return np.sum(np.tanh(a)) + para['lamda'] * np.trace(kk.dot(kk.T))


def jacobian(x, WDR, W, RR, a, para, Y):
    l = (1 - np.tanh(a)* np.tanh(a))* Y;
    grad = l + 2 * para['lamda'] * (x - WDR - W.T.dot(x.dot(RR.T)) + W.T.dot(WDR.dot(RR.T)))
    return grad


def steepest2(x0, WW, RR, WDR, a, Y, para):
    imax = 10
    # 函数的维度不确定
    # W = np.zeros((2, imax))
    # epo = np.zeros((2, imax))
    # W[:, 0] = x0
    i = 1
    x = x0
    grad = jacobian(x, WDR, WW, RR, a, para, Y)
    delta = sum(grad ** 2)  # 初始误差

    # f = open("最速.txt", 'w')

    while i < imax and (delta > 10 ** (-5)).all():
        p = -jacobian(x, WDR, WW, RR, a, para, Y)
        x0 = x
        alpha = goldsteinsearch(rosenbrock, jacobian, p, x, 1, 0.1, 2, WW, RR, WDR, a, Y, para)
        x = x + alpha * p
        # W[:, i] = x

        # epo[:, i] = np.array((i, delta))
        # f.write(str(i) + "        " + str(delta) + "\n")  #
        # print(i, np.array((i, delta)))
        grad = jacobian(x, WDR, WW, RR, a, para, Y)
        delta = sum(grad ** 2)
        i = i + 1
    # print("迭代次数为:", i)
    # print("近似最优解为:")
    # print(x, '\n')
    # W = W[:, 0:i]  # 记录迭代点
    return x, 1


# if __name__ == "__main__":
#     X1 = np.arange(-1.5, 1.5 + 0.05, 0.05)
#     X2 = np.arange(-3.5, 4 + 0.05, 0.05)
#     [x1, x2] = np.meshgrid(X1, X2)
#
#     f = 2 * x1 ** 2 + x2 ** 2
#     plt.contour(x1, x2, f, 20)  # 画出函数的20条轮廓线
#     x0 = np.array([1, 1])
#     list_out = steepest(x0)
#     W = list_out[0]
#     epo = list_out[1]
#     plt.plot(W[0, :], W[1, :], 'g*-')  # 画出迭代点收敛的轨迹
#     plt.show()