import numpy as np

def Ranking_loss(Outputs,test_target):
    num_class = Outputs.shape[0]
    num_instance = Outputs.shape[1]

    temp_Outputs = []
    temp_test_target = []

    for i in range(num_instance):
        temp = test_target[:, i]
        if ((np.sum(temp)!=num_class) and (np.sum(temp)!=-num_class)):
            if (len(temp_Outputs) != 0):
                temp_Outputs = np.c_[temp_Outputs, Outputs[:, i]]
            else:
                temp_Outputs = Outputs[:, i]
            if (len(temp_test_target) != 0):
                temp_test_target = np.c_[temp_test_target, temp]
            else:
                temp_test_target = temp

    Outputs = temp_Outputs
    test_target = temp_test_target
    num_class = Outputs.shape[0]
    num_instance = Outputs.shape[1]

    Label = {}
    not_Label = {}

    Label_size = np.zeros((1, num_instance))
    for i in range(num_instance):
        temp = test_target[:, i]
        Label_size[0, i] = np.sum(temp == np.ones(num_class))
        for j in range(num_class):
            if (temp[j] == 1):
                if(i in Label.keys()):
                    Label[i] = np.r_[Label[i], j]
                else:
                    Label[i] = np.array([j])
            else:
                if (i in not_Label.keys()):
                    not_Label[i] = np.r_[not_Label[i], j]
                else:
                    not_Label[i] = np.array([j])

    rankloss = 0
    rl_binary = []
    for i in range(num_instance):
        temp = 0
        for m in range(int(Label_size[0, i])):
            for n in range(int(num_class - Label_size[0, i])):
                if (Outputs[Label[i][m], i] <= Outputs[not_Label[i][n], i]):
                    temp = temp + 1
        t = Label_size[0, i] * (num_class - Label_size[0, i])
        rl_binary.append(temp / t)
        rankloss = rankloss + temp / t

    RankingLoss = rankloss / num_instance
    return RankingLoss

