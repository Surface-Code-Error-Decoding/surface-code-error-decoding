prob_array = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 
              0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 
              0.1 ]


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

    for instance in range(numinst):
        print("*"*40)
        print("*"*40)
        print(f"Physical Error Prob: {p} | Instance: {instance}")

        dataset = createData(pro, d, n, 
                            restricted, 
                            px = probX, 
                            py = probY, 
                            pz = probZ)


        X_test, y_pred, y_test = training_complex_CNN_model(dataset, d, 
                                                            split = split, 
                                                            epochs = epochs, 
                                                            batch_size = batch_size)
        

        logicalerror, logicalX, logicalZ, logicalY, acc_lld = postprocessing_equiv_logical_LOWLEVEL(X_test, 
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