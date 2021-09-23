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

    return data


