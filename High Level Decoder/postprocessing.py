import numpy as np

def postprocessing_lowlevel(qubitmatching, d, stabilizer_set_X_d3, stabilizer_set_Z_d3):

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


    logicalX = False 
    logicalZ = False
    logicalY = False
    correct_pred = False 


    if ((if_prediction_X_correct and if_prediction_Z_correct) or (if_stabilizer_X and if_stabilizer_Z) 
        or (if_prediction_X_correct and if_stabilizer_Z) or (if_stabilizer_X and if_prediction_Z_correct)):  # when ML model predicts the exact error or its a stabilizer
        
        correct_pred = True 


    else:

        qubitmatching_matrix_X = qubitmatching_X.reshape(d,d)
        qubitmatching_matrix_Z = qubitmatching_Z.reshape(d,d)
        qubitmatching_matrix_Z = np.transpose(qubitmatching_matrix_Z)

        logicalz=[[1,1,1, 0,0,0, 0,0,0],
                [0,0,0, 1,1,1, 0,0,0],
                [0,0,0, 0,0,0, 1,1,1],
                [0,0,1, 0,1,0, 1,0,0]]
        logicalz = np.array(logicalz)

        logicalx=[[1,0,0, 1,0,0, 1,0,0],
                [0,1,0, 0,1,0, 0,1,0],
                [0,0,1, 0,0,1, 0,0,1],
                [1,0,0, 0,1,0, 0,0,1]]
        logicalx = np.array(logicalx)


        ## logical-X
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
                    break


        ## logical-Z
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
                    break


        ## logical-Y
        if (logicalX and logicalZ):

            logicalY = True
            logicalX = False
            logicalZ = False


    return (correct_pred, logicalX, logicalZ, logicalY)




def postprocessing_highlevel(hld_pred, lld_pred):

    correct_pred_lld, logicalX_lld, logicalZ_lld, logicalY_lld = lld_pred

    identity_hld = False  
    logicalX_hld = False 
    logicalZ_hld = False 
    logicalY_hld = False 

    if (hld_pred == 0):  ## HLD predicts Identity
    
        identity_hld = False 
        logicalX_hld = logicalX_lld
        logicalZ_hld = logicalZ_lld
        logicalY_hld = logicalY_lld

    elif (hld_pred == 1):   ## HLD predicts X-error 

        if logicalX_lld: 
            identity_hld = True 
            logicalX_hld = False 
            logicalZ_hld = False 
            logicalY_hld = False 

        elif logicalZ_lld:
            identity_hld = False 
            logicalX_hld = False 
            logicalZ_hld = False 
            logicalY_hld = True 

        elif logicalY_lld: 
            identity_hld = False 
            logicalX_hld = False 
            logicalZ_hld = True 
            logicalY_hld = False  


    elif (hld_pred == 2):   ## HLD predicts Z-error

        if logicalX_lld: 
            identity_hld = False 
            logicalX_hld = False 
            logicalZ_hld = False 
            logicalY_hld = True 

        elif logicalZ_lld:
            identity_hld = True 
            logicalX_hld = False 
            logicalZ_hld = False 
            logicalY_hld = False 

        elif logicalY_lld: 
            identity_hld = False 
            logicalX_hld = True 
            logicalZ_hld = False 
            logicalY_hld = False 


    
    elif (hld_pred == 3):  ## HLD predicts Y-error

        if logicalX_lld: 
            identity_hld = False 
            logicalX_hld = False 
            logicalZ_hld = True 
            logicalY_hld = False 

        elif logicalZ_lld:
            identity_hld = False 
            logicalX_hld = True 
            logicalZ_hld = False 
            logicalY_hld = False 

        elif logicalY_lld: 
            identity_hld = True 
            logicalX_hld = False 
            logicalZ_hld = False 
            logicalY_hld = False 



    return (identity_hld, logicalX_hld, logicalZ_hld, logicalY_hld) 



def error_format_01to123(error_01, d):

    error_01 = error_01.reshape(d**2, 2)
    error_123 = np.zeros(d**2, dtype=float)

    for i in range(d**2):
        if (error_01[i][0]==1 and error_01[i][1]==0):
            error_123[i] = 1

        elif (error_01[i][0]==0 and error_01[i][1]==1):
            error_123[i] = 2

        elif (error_01[i][0]==1 and error_01[i][1]==1):
            error_123[i] = 3

    return error_123 


