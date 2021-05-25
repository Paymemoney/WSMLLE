import numpy as np

def MacroF1(Pre_Labels,test_target):
    # Computing the Macro_AUC
    (num_class, num_instance) = Pre_Labels.shape
    num_P_instance = np.zeros((num_class, 1))
    num_N_instance = np.zeros((num_class, 1))
    num_P_pre = np.zeros((num_class, 1))
    num_N_pre = np.zeros((num_class, 1))
    fm = np.zeros((num_class, 1))
    for i in range(num_class):
        num_P_instance[i, 0] = np.sum(test_target[i, :] == 1)
        num_N_instance[i, 0] = num_instance - num_P_instance[i, 0]
        num_P_pre[i, 0] = np.sum(Pre_Labels[i, :] == 1)
        num_N_pre[i, 0] = num_instance - num_P_pre[i, 0]

        num_P = np.sum(Pre_Labels[i, :] == 1)
        num_N = num_instance - num_P
        pre = Pre_Labels[i, :]
        pre0 = pre
        instance = test_target[i, :]
        pre0[pre0 == -1] = 0
        TP = np.sum(pre0 == instance)
        pre0 = pre
        pre0[pre0 == 1] = 0
        TN = np.sum(pre0 == instance)
        FP = num_P - TP
        FN = num_N - TN
        if (TP + FP + FN == 0):
            fm[i, 0] = 1
        else:
            fm[i, 0] = 2 * TP / (2 * TP + FN + FP)

    macrof1 = np.sum(fm) / num_class
    return macrof1


