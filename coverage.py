import numpy as np

def coverage(Outputs,test_target):
    num_class = Outputs.shape[0]
    num_instance = Outputs.shape[1]

    # temp_Outputs = np.array([])
    # temp_test_target = np.array([])
    #
    # for i in range(num_instance):
    #     temp = test_target[:, i]
    #     if ((np.sum(temp)!=num_class) and (np.sum(temp)!=-num_class)):
    #         if(len(temp_Outputs)!=0):
    #             temp_Outputs = np.concatenate((temp_Outputs, Outputs[:, i]), axis=1)
    #         else:
    #             temp_Outputs = Outputs[:, i]
    #         if (len(temp_test_target) != 0):
    #             temp_test_target = np.concatenate((temp_test_target, temp), axis=1)
    #         else:
    #             temp_test_target = temp
    #
    # Outputs = temp_Outputs
    # test_target = temp_test_target
    # num_class = Outputs.shape[0]
    # num_instance = Outputs.shape[1]

    Label = {}
    not_Label = {}

    Label_size = np.zeros((1, num_instance))
    for i in range(num_instance):
        temp = test_target[:, i]
        Label_size[0, i] = np.sum(temp == np.ones(num_class))
        for j in range(num_class):
            if (temp[j] == 1):
                if (i in Label.keys()):
                    Label[i] = np.c_[Label[i], j]
                else:
                    Label[i] = np.array([j])
            else:
                if (i in not_Label.keys()):
                    not_Label[i] = np.c_[not_Label[i], j]
                else:
                    not_Label[i] = np.array([j])

    cover = 0
    for i in range(num_instance):
        temp = Outputs[:, i]
        tempvalue = np.sort(temp)
        index = np.argsort(temp)
        temp_min = num_class + 1
        for m in range(int(Label_size[0, i])):
            if(Label[i].ndim==2):
                loc = np.argmax(Label[i][0,m] in index)
            else:
                loc = np.argmax(Label[i][m] in index)
            if (loc < temp_min):
                temp_min = loc

        cover = cover + (num_class - temp_min + 1)
    Coverage = (cover / num_instance) - 1
    return Coverage