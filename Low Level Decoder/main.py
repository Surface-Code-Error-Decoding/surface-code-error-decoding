import numpy as np

from data_generation import createData
from models import training_CNN_model, training_FFNN_model
from postprocessing import postprocessing_low_level_d3, postprocessing_low_level_d5
from stabilizer_set import generate_stabilizer_set 



prob_array = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 
              0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 
              0.1 ]



restricted_d3 = [[0,0], [0,2], [0,3], 
                [1,0], 
                [2,3],  
                [3,0], [3,1], [3,3]] 


restricted_d5 = [[0,0], [0,2], [0,4], [0,5], 
                [1,0], 
                [2,5], 
                [3,0], 
                [4,5],   
                [5,0], [5,1], [5,3], [5,5]]  


stabilizerX_d3 = [[0,1], [7,8], [1,2,4,5], [3,4,6,7]]

stabilizerZ_d3 = [[2,5], [3,6], [0,1,3,4], [4,5,7,8]]


stabilizerX_d5 = [[0,1], [2,3], [21,22], [23,24], 
                  [1,2,6,7], [3,4,8,9], [5,6,10,11], [7,8,12,13], 
                  [11,12,16,17], [13,14,18,19], [15,16,20,21], [17,18,22,23]]

stabilizerZ_d5 = [[4,9], [14,19], [5,10], [15,20], 
                  [0,1,5,6], [2,3,7,8], [6,7,11,12], [8,9,13,14], 
                  [10,11,15,16], [12,13,17,18], [16,17,21,22], [18,19,23,24]] 



############################
logicalerror_lld_list = []
logicalX_lld_list = []
logicalZ_lld_list = []
logicalY_lld_list = []
acc_lld_list = []
############################

for i in range(0, len(prob_array)):

    p = prob_array[i]

    print("\n\n")
    print("#"*40)
    print("#"*40)
    
    print(f"Physical qubit error: {p}")

    sum_logicalerror = 0
    sum_logicalX = 0
    sum_logicalZ = 0
    sum_logicalY = 0
    sum_acc_lld = 0

    pro = 1-pow((1-p),8)

    ###################################
    ## ----------Tune these------------
    d = 3
    n = 100000
    numinst = 5
    split = 0.3
    epochs = 1000
    batch_size = 10000

    probX = 1/3
    probY = 1/3
    probZ = 1/3
    ## --------------------------------
    ###################################


    if(d==3):
        restricted = restricted_d3
    elif(d==5):
        restricted = restricted_d5


    for instance in range(numinst):
        print("*"*40)
        print("*"*40)
        print(f"Physical Error Prob: {p} | Instance: {instance}")

        dataset = createData(pro, d, n, 
                            restricted, 
                            px = probX, 
                            py = probY, 
                            pz = probZ)


        


        X_test, y_pred, y_test = training_FFNN_model(dataset, d, 
                                                    split = split, 
                                                    epochs = epochs, 
                                                    batch_size = batch_size)
        

        if(d==3):

            stabilizer_set_X = generate_stabilizer_set(stabilizerX_d3) 
            stabilizer_set_Z = generate_stabilizer_set(stabilizerZ_d3)

            logicalerror, logicalX, logicalZ, logicalY, acc_lld = postprocessing_low_level_d3(X_test, 
                                                                                                y_pred, 
                                                                                                y_test, 
                                                                                                d, 
                                                                                                stabilizer_set_X,   
                                                                                                stabilizer_set_Z) 


        elif(d==5):

            stabilizer_set_X = generate_stabilizer_set(stabilizerX_d5) 
            stabilizer_set_Z = generate_stabilizer_set(stabilizerZ_d5) 

            logicalerror, logicalX, logicalZ, logicalY, acc_lld = postprocessing_low_level_d5(X_test, 
                                                                                                y_pred, 
                                                                                                y_test, 
                                                                                                d, 
                                                                                                stabilizer_set_X,   
                                                                                                stabilizer_set_Z) 


        


        sum_logicalerror += logicalerror 
        sum_logicalX += logicalX 
        sum_logicalZ += logicalZ 
        sum_logicalY += logicalY 
        sum_acc_lld += acc_lld

    
    avg_logicalerror = sum_logicalerror / numinst
    avg_logicalX = sum_logicalX / numinst
    avg_logicalZ = sum_logicalZ / numinst
    avg_logicalY = sum_logicalY / numinst
    avg_acc_lld = sum_acc_lld / numinst

    logicalerror_lld_list.append(avg_logicalerror)
    logicalX_lld_list.append(avg_logicalX)
    logicalZ_lld_list.append(avg_logicalZ)
    logicalY_lld_list.append(avg_logicalY)
    acc_lld_list.append(avg_acc_lld)


    print("-"*40)
    print("-"*40)



file_name = "Accuracy_Error_" + str(prob_array[0]) + "-" + str(prob_array[-1]) + "_complex-CNN_d=" + str(d) + ".csv"

with open(file_name, "ab") as f:
    np.savetxt(f, np.array([prob_array, 
                            logicalerror_lld_list, 
                            logicalX_lld_list, 
                            logicalZ_lld_list, 
                            logicalY_lld_list, 
                            acc_lld_list]),  delimiter=",", fmt="%f")