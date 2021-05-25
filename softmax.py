import numpy as np

def softmax(x, axis=1):
    # 计算每行的最大值
    # row_max = x.max(axis=axis)
    #
    # # 每行元素都需要减去对应的最大值，否则求exp(x)会溢出，导致inf情况
    # row_max = row_max.reshape(-1, 1)
    # x = x - row_max

    # 计算e的指数次幂
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    s = x_exp / x_sum
    return s