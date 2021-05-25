import numpy as np

def One_error(Outputs,test_target):
    num_class = Outputs.shape[0]
    num_instance = Outputs.shape[1]

    temp_Outputs = np.array([])
    temp_test_target = np.array([])

    for i in range(num_instance):
        temp = test_target[:, i]
        if ((np.sum(temp)!=num_class) and (np.sum(temp)!=-num_class)):
            if(len(temp_Outputs)!=0):
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
        Label_size[0, i] = np.sum(temp == np.ones((num_class, 1)))
        for j in range(num_class):
            if (temp[j] == 1):
                if (i in Label.keys()):
                    Label[i] = np.r_[Label[i], j]
                else:
                    Label[i] = j
            else:
                if (i in not_Label.keys()):
                    not_Label[i] = np.r_[not_Label[i], j]
                else:
                    not_Label[i] = j

    oneerr = 0
    for i in range(num_instance):
        indicator = 0
        temp = Outputs[:, i]
        maximum = np.max(temp)
        for j in range(num_class):
            if (temp[j] == maximum):
                if (np.sum(Label[i]==j)):
                    indicator = 1
                    break
        if(indicator==0):
            oneerr = oneerr + 1
    OneError = oneerr / num_instance
    return OneError