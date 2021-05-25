import numpy as np

def MicroF1(Pre_Labels,test_target):
    # Computing the Macro_AUC
    (num_class, num_instance) = Pre_Labels.shape
    num_P_instance = np.zeros((num_class, 1))
    num_N_instance = np.zeros((num_class, 1))
    count_valid_label = 0
    fm = np.zeros((num_class, 1))
    sumTP = 0
    sumTN = 0
    sumFP = 0
    sumFN = 0
    for i in range(num_class):
        num_P_instance[i, 0] = np.sum(test_target[i, :] == 1)
        num_N_instance[i, 0] = num_instance - num_P_instance[i, 0]
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
        sumTP = sumTP + TP
        sumTN = sumTN + TN
        sumFP = sumFP + FP
        sumFN = sumFN + FN

    microf1 = 2 * sumTP / (2 * sumTP + sumFN + sumFP)
    return microf1


