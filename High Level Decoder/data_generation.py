import numpy as np
import random 
import csv 


def generateError(p, d, px, py, pz):

    qubiterror = np.zeros(d**2, dtype=int)
    numrange = 1e7

    for pos in range(0, d**2):
        num1 = random.randint(1, numrange)
        if (num1 <= p*numrange):
            err_type = random.randint(1,10000)
            if (err_type <= px*10000):
                qubiterror[pos] = 1
            elif ((err_type <= px*10000+pz*10000) and (err_type > px*10000)):
                qubiterror[pos] = 2
            else:
                qubiterror[pos] = 3

    return qubiterror 


def LLDerrorToHLDerror(qubiterror, d, stabilizer_set_X_d3, stabilizer_set_Z_d3):

    qubiterror = qubiterror.reshape(d**2, 2)
    qubiterror_X = qubiterror[:,0]
    qubiterror_Z = qubiterror[:,1]

    if_prediction_X_correct = np.bitwise_xor(qubiterror_X,1).all()
    if_prediction_Z_correct = np.bitwise_xor(qubiterror_Z,1).all()

    if_stabilizer_X = False
    if_stabilizer_Z = False

    nonzero_index_X = []
    for index in range(len(qubiterror_X)):
        if (qubiterror_X[index] == 1):
            nonzero_index_X.append(index)

    if nonzero_index_X in stabilizer_set_X_d3:
        if_stabilizer_X = True
    

    nonzero_index_Z = []
    for index in range(len(qubiterror_Z)):
        if (qubiterror_Z[index] == 1):
            nonzero_index_Z.append(index)
    
    if nonzero_index_Z in stabilizer_set_Z_d3:
        if_stabilizer_Z = True


    if ((if_prediction_X_correct and if_prediction_Z_correct) or (if_stabilizer_X and if_stabilizer_Z) 
        or (if_prediction_X_correct and if_stabilizer_Z) or (if_stabilizer_X and if_prediction_Z_correct)):  

        return np.array([0])  #Identity

    else: 

        qubiterror_matrix_X = qubiterror_X.reshape(d,d)
        qubiterror_matrix_Z = qubiterror_Z.reshape(d,d)
        qubiterror_matrix_Z = np.transpose(qubiterror_matrix_Z)

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



        ##LogicalX check
        logicalX = False
        
        if (np.count_nonzero(qubiterror_matrix_X[0] == 1) > 0 
            and np.count_nonzero(qubiterror_matrix_X[1] == 1) > 0 
            and np.count_nonzero(qubiterror_matrix_X[2] == 1) > 0):
        
            for i in range(len(logicalz)):
                commutecheck = np.bitwise_and(qubiterror_X, logicalz[i]) 
                commutecount = 0
                for c_i  in range (len(commutecheck)):
                    if (commutecheck[c_i]==1):
                        commutecount = commutecount+1

                if ((commutecount%2)!=0):
                    logicalX = True
                    break
        
        ##LogicalZ check
        logicalZ = False

        if (np.count_nonzero(qubiterror_matrix_Z[0] == 1) > 0 
            and np.count_nonzero(qubiterror_matrix_Z[1] == 1) > 0 
            and np.count_nonzero(qubiterror_matrix_Z[2] == 1) > 0):
        
            for i in range(len(logicalx)):
                commutecheck = np.bitwise_and(qubiterror_Z, logicalx[i]) 
                commutecount = 0
                for c_i  in range (len(commutecheck)):
                    if (commutecheck[c_i]==1):
                        commutecount = commutecount+1

                if ((commutecount%2)!=0):
                    logicalZ = True
                    break


        ##LogicalY check
        logicalY = False
        if (logicalX and logicalZ):
            logicalY = True
            logicalX = False
            logicalZ = False


        if logicalX:
            return np.array([1])  # X-error

        if logicalZ:
            return np.array([2])  # Z-error

        if logicalY:
            return np.array([3])  # Y-error

        return np.array([0])




def errorToSyndrome(qubiterror, restricted, d):

    if(d==3):
        filepath = 'databaseautomate_d-3.csv'
    elif(d==5):
        filepath = 'databaseautomate_d-5.csv'
    elif(d==7):
        filepath = 'databaseautomate_d-7.csv'
    else:
        print("Database automate file out of bound")
    

    traininglabel = np.zeros(2*(d**2), dtype=int)

    trainingrowX = np.zeros((d+1)**2, dtype=int)
    trainingrowY = np.zeros((d+1)**2, dtype=int)
    trainingrowZ = np.zeros((d+1)**2, dtype=int)

    
    out_arr = np.zeros((d+1)**2, dtype=int)

    for index in range(d**2):
        
        # X error
        if (qubiterror[index] == 1):
            traininglabel[2*index] = 1

            qubiterrorX = np.zeros(d**2, dtype=int)
            qubiterrorX[index] = 1

            with open(filepath) as csvDataFile:
                csvReaderData = csv.reader(csvDataFile)

                for bitflip_row in csvReaderData:
                    rowdata = np.array(bitflip_row).astype(int)
                    if ((rowdata[0:d**2]==qubiterrorX).all(axis=0)):
                        break
                
                trainingrowXpart = np.array(rowdata[d**2:(d**2)+((d+1)**2)]).flatten().astype(int)
                trainingrowX = np.bitwise_xor(trainingrowX, trainingrowXpart).astype(int)



        # Z error
        elif (qubiterror[index] == 2):
            traininglabel[(2*index)+1] = 1

            qubiterrorZ = np.zeros(d**2, dtype=int)
            qubiterrorZ[index] = 2

            with open(filepath) as csvDataFile:
                csvReaderData = csv.reader(csvDataFile)

                for phaseflip_row in csvReaderData:
                    rowdata = np.array(phaseflip_row).astype(int)
                    if ((rowdata[0:d**2]==qubiterrorZ).all(axis=0)):
                        break
                
                trainingrowZpart = np.array(rowdata[d**2:(d**2)+((d+1)**2)]).flatten().astype(int)
                trainingrowZ = np.bitwise_xor(trainingrowZ, trainingrowZpart).astype(int)



        # Y error
        elif (qubiterror[index] == 3):
            traininglabel[2*index] = 1
            traininglabel[(2*index)+1] = 1

            qubiterrorY = np.zeros(d**2, dtype=int)
            qubiterrorY[index] = 3

            with open(filepath) as csvDataFile:
                csvReaderData = csv.reader(csvDataFile)

                for bitphaseflip_row in csvReaderData:
                    rowdata = np.array(bitphaseflip_row).astype(int)
                    if ((rowdata[0:d**2]==qubiterrorY).all(axis=0)):
                        break
                
                trainingrowYpart = np.array(rowdata[d**2:(d**2)+((d+1)**2)]).flatten().astype(int)
                trainingrowY = np.bitwise_xor(trainingrowY, trainingrowYpart).astype(int)


    out_arr = np.bitwise_xor(trainingrowX, trainingrowZ)
    out_arr = np.bitwise_xor(out_arr, trainingrowY).flatten().astype(int)

    traininglabel = traininglabel.flatten().astype(int)

    trainingrow = np.concatenate((traininglabel, out_arr))

    #### Check if in the syndrome any restricted bit or phase got flipped ####

    syndrome = np.copy(trainingrow[2*(d**2) : 2*(d**2)+((d+1)**2)])
    syndrome = syndrome.reshape(d+1,d+1)
    for res in restricted:
        if(syndrome[res[0], res[1]] == 1):
            print("Dummy Node can't show syndrome")

    #################################################################

    return trainingrow



def createData(p, d, n, restricted, px, py, pz):

    data = []
    for _ in range(n):
        qubiterror = generateError(p, d, px, py, pz)
        trainingrow = errorToSyndrome(qubiterror, restricted, d)
        data.append(trainingrow)

    data = np.array(data)

    print(f"Dataset Size: {data.shape}") 

    #############################
    # print(list(set(data[:,-1])))

    if (len(list(set(data[:,-1]))) == 4):
        print("All 4 HLD error is present")
    else:
        r = np.array([0,1,2,3])
        a = np.array(list(set(data[:,-1]))) 

        absent = []
        for i in r:
            if not i in a:
                absent.append(i) 

        fill = random.sample(range(0,4), len(absent)) 
        for i in range(len(absent)):
            data[fill[i]][-1] = absent[i] 

        if (len(list(set(data[:,-1]))) == 4):
            print(f"Randomly filled {absent} at {fill} as dummy")

    # print(list(set(data[:,-1])))

    #############################

    return data 


    