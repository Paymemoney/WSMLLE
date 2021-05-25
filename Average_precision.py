import numpy as np

def Average_precision(Outputs,test_target):
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
                temp_test_target = np.c_[temp_test_target, Outputs[:, i]]
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
                if (i in Label.keys()):
                    Label[i] = np.r_[Label[i], j]
                else:
                    Label[i] = np.array([j])
            else:
                if (i in not_Label.keys()):
                    not_Label[i] = np.r_[not_Label[i], j]
                else:
                    not_Label[i] = np.array([j])

    aveprec = 0
    for i in range(num_instance):
        temp = Outputs[:, i]
        tempvalue = np.sort(temp)
        index = np.argsort(temp)
        indicator = np.zeros((1, num_class))
        for m in range(int(Label_size[0, i])):
            if (Label[i].ndim == 2):
                loc = np.argmax(Label[i][0, m] in index)
            else:
                loc = np.argmax(Label[i][m] in index)
                indicator[0, loc] = 1
        summary = 0
        for m in range(int(Label_size[0, i])):
            if (Label[i].ndim == 2):
                loc = np.argmax(Label[i][0, m] in index)
            else:
                loc = np.argmax(Label[i][m] in index)
            summary=summary+np.sum(indicator[loc:num_class])/(num_class-loc+1)
        aveprec = aveprec + summary / Label_size[0, i]

    Average_Precision=aveprec/num_instance
    return Average_Precision