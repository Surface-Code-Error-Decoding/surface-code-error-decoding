import numpy as np



def postprocessing_low_level_d3(X_test, y_pred, y_test, d, stabilizer_set_X_d3, stabilizer_set_Z_d3):

    total = 0
    correct = 0
    logicalerrorcount = 0
    logicalXcount = 0
    logicalZcount = 0
    logicalYcount = 0
    
    ##confirming data length
    if (X_test.shape[0]!=y_pred.shape[0]) or (X_test.shape[0]!=y_test.shape[0]):
        print("Data length not equal")

    data_len = X_test.shape[0]

    for index in range(data_len):


        total += 1

        predictedqubit = y_pred[index].flatten().astype(int)
        truequbit = y_test[index].flatten().astype(int)

        qubitmatching = np.bitwise_xor(truequbit, predictedqubit)
        qubitmatching = qubitmatching.reshape(d**2, 2)
        qubitmatching_X = qubitmatching[:,0]
        qubitmatching_Z = qubitmatching[:,1]

        if_prediction_X_correct = np.bitwise_xor(qubitmatching_X,1).all()
        if_prediction_Z_correct = np.bitwise_xor(qubitmatching_Z,1).all()

        if_stabilizer_X = False
        if_stabilizer_Z = False

        nonzero_index_X = []
        for index in range(len(qubitmatching_X)):
            if (qubitmatching_X[index] == 1):
                nonzero_index_X.append(index)

        if nonzero_index_X in stabilizer_set_X_d3:
            if_stabilizer_X = True
        

        nonzero_index_Z = []
        for index in range(len(qubitmatching_Z)):
            if (qubitmatching_Z[index] == 1):
                nonzero_index_Z.append(index)
        
        if nonzero_index_Z in stabilizer_set_Z_d3:
            if_stabilizer_Z = True 




        
        if ((if_prediction_X_correct and if_prediction_Z_correct) or (if_stabilizer_X and if_stabilizer_Z) or (if_prediction_X_correct and if_stabilizer_Z) or (if_stabilizer_X and if_prediction_Z_correct)):  # when ML model predicts the exact error or its a stabilizer
            correct += 1

        else: # when prediction not correct, need to check if logical error
        

            qubitmatching_matrix_X = qubitmatching_X.reshape(d,d)
            qubitmatching_matrix_Z = qubitmatching_Z.reshape(d,d)
            qubitmatching_matrix_Z = np.transpose(qubitmatching_matrix_Z)

            logicalz = [[1,1,1, 0,0,0, 0,0,0], 
                        [0,0,0, 1,1,1, 0,0,0],
                        [0,0,0, 0,0,0, 1,1,1],
                        [0,0,1, 0,1,0, 1,0,0]]
            logicalz = np.array(logicalz)

            logicalx = [[1,0,0, 1,0,0, 1,0,0],
                        [0,1,0, 0,1,0, 0,1,0],
                        [0,0,1, 0,0,1, 0,0,1],
                        [1,0,0, 0,1,0, 0,0,1]]
            logicalx = np.array(logicalx)


            logicalX = False
            
            if (np.count_nonzero(qubitmatching_matrix_X[0] == 1) > 0 
                and np.count_nonzero(qubitmatching_matrix_X[1] == 1) > 0 
                and np.count_nonzero(qubitmatching_matrix_X[2] == 1) > 0):
                
                for i in range(len(logicalz)):
                    commutecheck = np.bitwise_and(qubitmatching_X, logicalz[i]) 
                    commutecount = 0
                    for c_i  in range (len(commutecheck)):
                        if (commutecheck[c_i]==1):
                            commutecount = commutecount+1

                    if ((commutecount%2)!=0):
                        logicalX=True
                        logicalXcount += 1
                        logicalerrorcount += 1
                        break
            
            ####################################
            
            logicalZ = False

            if (np.count_nonzero(qubitmatching_matrix_Z[0] == 1) > 0 
                and np.count_nonzero(qubitmatching_matrix_Z[1] == 1) > 0 
                and np.count_nonzero(qubitmatching_matrix_Z[2] == 1) > 0):
                
                for i in range(len(logicalx)):
                    commutecheck = np.bitwise_and(qubitmatching_Z, logicalx[i]) 
                    commutecount = 0
                    for c_i  in range (len(commutecheck)):
                        if (commutecheck[c_i]==1):
                            commutecount = commutecount+1

                    if ((commutecount%2)!=0):
                        logicalZ=True
                        logicalZcount += 1
                        logicalerrorcount += 1
                        break


            ##LogicalY check
            if (logicalX and logicalZ):
                logicalY = True
                logicalYcount += 1

                logicalX = False
                logicalZ = False
                logicalXcount -= 1
                logicalZcount -= 1

                logicalerrorcount -= 1

    print(f"       Total cases => {total}")
    print(f"    Correct by LLD => {correct}")
    print(f"Not correct by LLD => {total-correct}")
    print(f"ML model accuracy  => {(correct/total)*100} %")
    print("-"*40)

    print(f"    Logical X count => {logicalXcount}")
    print(f"    Logical Z count => {logicalZcount}")
    print(f"    Logical Y count => {logicalYcount}")
    print(f"Logical Error count => {logicalerrorcount}")
    print(f"Logical Error rate  => {logicalerrorcount / total}")
    print("-"*40)
    
    return (logicalerrorcount / total, 
            logicalXcount / total, 
            logicalZcount / total, 
            logicalYcount / total, 
            (correct/total)*100)







def postprocessing_low_level_d5(X_test, y_pred, y_test, d, stabilizer_set_X_d5, stabilizer_set_Z_d5):

    total = 0
    correct = 0
    logicalerrorcount = 0
    logicalXcount = 0
    logicalZcount = 0
    logicalYcount = 0 
    

    ##confirming data length
    if (X_test.shape[0]!=y_pred.shape[0]) or (X_test.shape[0]!=y_test.shape[0]):
        print("Data length not equal")

    data_len = X_test.shape[0]

    for index in range(data_len):

        total += 1

        predictedqubit = y_pred[index].flatten().astype(int)
        truequbit = y_test[index].flatten().astype(int)

        qubitmatching = np.bitwise_xor(truequbit, predictedqubit)
        qubitmatching = qubitmatching.reshape(d**2, 2)
        qubitmatching_X = qubitmatching[:,0]
        qubitmatching_Z = qubitmatching[:,1]

        if_prediction_X_correct = np.bitwise_xor(qubitmatching_X,1).all()
        if_prediction_Z_correct = np.bitwise_xor(qubitmatching_Z,1).all()

        if_stabilizer_X = False
        if_stabilizer_Z = False

        nonzero_index_X = []
        for index in range(len(qubitmatching_X)):
            if (qubitmatching_X[index] == 1):
                nonzero_index_X.append(index)

        if nonzero_index_X in stabilizer_set_X_d5:
            if_stabilizer_X = True
        

        nonzero_index_Z = []
        for index in range(len(qubitmatching_Z)):
            if (qubitmatching_Z[index] == 1):
                nonzero_index_Z.append(index)
        
        if nonzero_index_Z in stabilizer_set_Z_d5:
            if_stabilizer_Z = True




        
        if ((if_prediction_X_correct and if_prediction_Z_correct) or (if_stabilizer_X and if_stabilizer_Z) or (if_prediction_X_correct and if_stabilizer_Z) or (if_stabilizer_X and if_prediction_Z_correct)):  # when ML model predicts the exact error or its a stabilizer
            correct += 1

        else: # when prediction not correct, need to check if logical error
        

            qubitmatching_matrix_X = qubitmatching_X.reshape(d,d)
            qubitmatching_matrix_Z = qubitmatching_Z.reshape(d,d)
            qubitmatching_matrix_Z = np.transpose(qubitmatching_matrix_Z)

            logicalz = [[1,1,1,1,1,  0,0,0,0,0,  0,0,0,0,0,  0,0,0,0,0,  0,0,0,0,0],
                        [0,0,0,0,0,  1,1,1,1,1,  0,0,0,0,0,  0,0,0,0,0,  0,0,0,0,0],
                        [0,0,0,0,0,  0,0,0,0,0,  1,1,1,1,1,  0,0,0,0,0,  0,0,0,0,0],
                        [0,0,0,0,0,  0,0,0,0,0,  0,0,0,0,0,  1,1,1,1,1,  0,0,0,0,0],
                        [0,0,0,0,0,  0,0,0,0,0,  0,0,0,0,0,  0,0,0,0,0,  1,1,1,1,1],
                        [0,0,0,0,1,  0,0,0,1,0,  0,0,1,0,0,  0,1,0,0,0,  1,0,0,0,0]]

            logicalz = np.array(logicalz)


            logicalx = [[1,0,0,0,0,  1,0,0,0,0,  1,0,0,0,0,  1,0,0,0,0,  1,0,0,0,0], 
                        [0,1,0,0,0,  0,1,0,0,0,  0,1,0,0,0,  0,1,0,0,0,  0,1,0,0,0], 
                        [0,0,1,0,0,  0,0,1,0,0,  0,0,1,0,0,  0,0,1,0,0,  0,0,1,0,0], 
                        [0,0,0,1,0,  0,0,0,1,0,  0,0,0,1,0,  0,0,0,1,0,  0,0,0,1,0], 
                        [0,0,0,0,1,  0,0,0,0,1,  0,0,0,0,1,  0,0,0,0,1,  0,0,0,0,1], 
                        [1,0,0,0,0,  0,1,0,0,0,  0,0,1,0,0,  0,0,0,1,0,  0,0,0,0,1]]

            logicalx = np.array(logicalx)


            logicalX = False
            
            if (np.count_nonzero(qubitmatching_matrix_X[0] == 1) > 0 
                and np.count_nonzero(qubitmatching_matrix_X[1] == 1) > 0 
                and np.count_nonzero(qubitmatching_matrix_X[2] == 1) > 0 
                and np.count_nonzero(qubitmatching_matrix_X[3] == 1) > 0 
                and np.count_nonzero(qubitmatching_matrix_X[4] == 1) > 0):
                
                for i in range(len(logicalz)):
                    commutecheck = np.bitwise_and(qubitmatching_X, logicalz[i]) 
                    commutecount = 0
                    for c_i  in range (len(commutecheck)):
                        if (commutecheck[c_i]==1):
                            commutecount = commutecount+1

                    if ((commutecount%2)!=0):
                        logicalX=True
                        logicalXcount += 1
                        logicalerrorcount += 1
                        break
            
            ####################################
            
            logicalZ = False

            if (np.count_nonzero(qubitmatching_matrix_Z[0] == 1) > 0 
                and np.count_nonzero(qubitmatching_matrix_Z[1] == 1) > 0 
                and np.count_nonzero(qubitmatching_matrix_Z[2] == 1) > 0 
                and np.count_nonzero(qubitmatching_matrix_Z[3] == 1) > 0 
                and np.count_nonzero(qubitmatching_matrix_Z[4] == 1) > 0):
                
                for i in range(len(logicalx)):
                    commutecheck = np.bitwise_and(qubitmatching_Z, logicalx[i]) 
                    commutecount = 0
                    for c_i  in range (len(commutecheck)):
                        if (commutecheck[c_i]==1):
                            commutecount = commutecount+1

                    if ((commutecount%2)!=0):
                        logicalZ=True
                        logicalZcount += 1
                        logicalerrorcount += 1
                        break


            ##LogicalY check
            if (logicalX and logicalZ):
                logicalY = True
                logicalYcount += 1

                logicalX = False
                logicalZ = False
                logicalXcount -= 1
                logicalZcount -= 1

                logicalerrorcount -= 1

    print(f"       Total cases => {total}")
    print(f"    Correct by LLD => {correct}")
    print(f"Not correct by LLD => {total-correct}")
    print(f"ML model accuracy  => {(correct/total)*100} %")
    print("-"*40)

    print(f"    Logical X count => {logicalXcount}")
    print(f"    Logical Z count => {logicalZcount}")
    print(f"    Logical Y count => {logicalYcount}")
    print(f"Logical Error count => {logicalerrorcount}")
    print(f"Logical Error rate  => {logicalerrorcount / total}")
    print("-"*40)
    
    return (logicalerrorcount / total, 
            logicalXcount / total, 
            logicalZcount / total, 
            logicalYcount / total, 
            (correct/total)*100)

