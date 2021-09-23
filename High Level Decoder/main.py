import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_theme()

from data_generation import createData, errorToSyndrome
from stabilizer_set import generate_stabilizer_set
from models import training_FFNN_model_HLD, training_FFNN_model_LLD 
from postprocessing import postprocessing_highlevel, postprocessing_lowlevel, error_format_01to123



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




#############################
logicalerror_lld_list = []
logicalX_lld_list = []
logicalZ_lld_list = []
logicalY_lld_list = []
acc_lld_list = []

logicalerror_hld_list = []
logicalX_hld_list = []
logicalZ_hld_list = []
logicalY_hld_list = []
acc_hld_list = []
#############################


for j in range(0, len(prob_array)):

    p = prob_array[j]

    print("\n\n")
    print("#"*40)
    print("#"*40)
    
    print(f"Physical qubit error: {p}")

    sum_logicalerror_count_lld = 0
    sum_logicalX_count_lld = 0
    sum_logicalZ_count_lld = 0
    sum_logicalY_count_lld = 0
    sum_acc_count_lld = 0 

    sum_logicalerror_count_hld = 0
    sum_logicalX_count_hld = 0
    sum_logicalZ_count_hld = 0
    sum_logicalY_count_hld = 0
    sum_acc_count_hld = 0

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
        stabilizer_set_X = generate_stabilizer_set(stabilizerX_d3) 
        stabilizer_set_Z = generate_stabilizer_set(stabilizerZ_d3) 


    for instance in range(numinst):
        print("*"*40)
        print()
        print(f"Physical Error Prob: {p} | Instance: {instance}")


        ## Data generation
        test_len = int(n*split)
        train_len = n-test_len

        print("-"*40)
        print()
        dataset_train = createData(pro, d, 
                                100000, 
                                restricted, 
                                px = probX, 
                                py = probY, 
                                pz = probZ)
        
        print()
        dataset_test = createData(pro, d, 
                                test_len, 
                                restricted, 
                                px = probX, 
                                py = probY, 
                                pz = probZ)
    
        

        ## Data splitting
        X_train_lld = dataset_train[0:train_len, 2*(d**2) : 2*(d**2)+((d+1)**2)].astype(float) 
        X_test_lld = dataset_test[:, 2*(d**2) : 2*(d**2)+((d+1)**2)].astype(float) 


        y_train_lld = dataset_train[0:train_len, 0 : 2*(d**2)]
        y_test_lld = dataset_test[:, 0 : 2*(d**2)] 

        y_train_hld = dataset_train[:, -1] 


        count = min(np.count_nonzero(y_train_hld==0), 
                    np.count_nonzero(y_train_hld==1), 
                    np.count_nonzero(y_train_hld==2), 
                    np.count_nonzero(y_train_hld==3))

        Icount = 0
        Xcount = 0
        Zcount = 0
        Ycount = 0

        X_train_hld = []
        y_train_hld = []

        for i in range(len(dataset_train)): 
            hld = dataset_train[i][-1] 
            
            if (hld==0):
                if Icount<count:
                    Icount += 1
                    X_train_hld.append(dataset_train[i][2*(d**2) : 2*(d**2)+((d+1)**2)].astype(float)) 
                    y_train_hld.append(hld) 

            elif (hld==1):
                if Xcount<count:
                    Xcount += 1
                    X_train_hld.append(dataset_train[i][2*(d**2) : 2*(d**2)+((d+1)**2)].astype(float)) 
                    y_train_hld.append(hld) 

            elif (hld==2):
                if Zcount<count:
                    Zcount += 1
                    X_train_hld.append(dataset_train[i][2*(d**2) : 2*(d**2)+((d+1)**2)].astype(float))
                    y_train_hld.append(hld) 

            elif (hld==3):
                if Ycount<count:
                    Ycount += 1
                    X_train_hld.append(dataset_train[i][2*(d**2) : 2*(d**2)+((d+1)**2)].astype(float))
                    y_train_hld.append(hld) 

        X_train_hld = np.array(X_train_hld)
        y_train_hld = np.array(y_train_hld) 


        print("-"*40)
        print()
        print(f"Each HLD label count => {count}")




        ## training both LLD and HLD model
        model_lld = training_FFNN_model_LLD(X_train_lld, y_train_lld, 
                                                d, 
                                                epochs = epochs, 
                                                batch_size = batch_size)
        
        model_hld = training_FFNN_model_HLD(X_train_hld, y_train_hld, 
                                                d, 
                                                epochs = epochs, 
                                                batch_size = batch_size)
        

        


        logicalerror_count_lld = 0
        logicalX_count_lld = 0
        logicalZ_count_lld = 0
        logicalY_count_lld = 0 
        acc_count_lld = 0

        logicalerror_count_hld = 0
        logicalX_count_hld = 0
        logicalZ_count_hld = 0
        logicalY_count_hld = 0
        acc_count_hld = 0


        test_len_hld = 0
        corrected_by_hld = 0


        for index in range(test_len):
        
            syndrome = X_test_lld[index] 
            truequbit = y_test_lld[index] 
            predictedqubit = model_lld.predict(np.array([syndrome])).flatten().astype(int) 

            qubitmatching = np.bitwise_xor(truequbit, predictedqubit) 

            correct_pred_lld, logicalX_lld, logicalZ_lld, logicalY_lld = postprocessing_lowlevel(qubitmatching, d, stabilizer_set_X, stabilizer_set_Z) 


            if correct_pred_lld:
                acc_count_lld += 1

            elif logicalX_lld:
                logicalX_count_lld += 1 
                logicalerror_count_lld += 1

            elif logicalZ_lld:
                logicalZ_count_lld += 1 
                logicalerror_count_lld += 1 

            elif logicalY_lld:
                logicalY_count_lld += 1 
                logicalerror_count_lld += 1


            


            if (logicalX_lld or logicalZ_lld or logicalY_lld):

                test_len_hld += 1

                qubitmatching_123 = error_format_01to123(qubitmatching, d) 
                syndromematching = errorToSyndrome(qubitmatching_123, restricted, d)
                syndromematching = syndromematching[2*(d**2) : 2*(d**2)+((d+1)**2)] 

                hld_pred = model_hld.predict(np.array([syndromematching])).flatten()
                hld_pred = np.argmax(hld_pred) 

                identity_hld, logicalX_hld, logicalZ_hld, logicalY_hld = postprocessing_highlevel(hld_pred, 
                                                                        (correct_pred_lld, logicalX_lld, logicalZ_lld, logicalY_lld)) 
                
            
                if identity_hld:
                    corrected_by_hld += 1

            else: 
                identity_hld = True 
                logicalX_hld = False
                logicalZ_hld = False
                logicalY_hld = False

            
            


            if identity_hld:
                acc_count_hld += 1

            elif logicalX_hld:
                logicalX_count_hld += 1 
                logicalerror_count_hld += 1

            elif logicalZ_hld:
                logicalZ_count_hld += 1 
                logicalerror_count_hld += 1 

            elif logicalY_hld:
                logicalY_count_hld += 1 
                logicalerror_count_hld += 1
        
        if (logicalerror_count_lld != test_len_hld):
            print("All LLD logical errors haven't gone to HLD") 


        sum_logicalerror_count_lld += logicalerror_count_lld / test_len 
        sum_logicalX_count_lld += logicalX_count_lld / test_len 
        sum_logicalZ_count_lld += logicalZ_count_lld / test_len 
        sum_logicalY_count_lld += logicalY_count_lld / test_len 
        sum_acc_count_lld += (acc_count_lld / test_len )*100 

        sum_logicalerror_count_hld += logicalerror_count_hld / test_len 
        sum_logicalX_count_hld += logicalX_count_hld / test_len 
        sum_logicalZ_count_hld += logicalZ_count_hld / test_len 
        sum_logicalY_count_hld += logicalY_count_hld / test_len 
        sum_acc_count_hld += (corrected_by_hld/test_len_hld)*100 

        print("-"*40)
        print()
        print("-------LLD Performance-------")
        print(f"           Total cases => {test_len}")
        print(f"        Correct by LLD => {acc_count_lld}")
        print(f"    Not correct by LLD => {test_len - acc_count_lld}")
        print(f" LLD ML model accuracy => {(acc_count_lld/test_len)*100} %")
        
        print()

        print(f"Logical Error count => {logicalerror_count_lld}")
        print(f"    Logical X count => {logicalX_count_lld}")
        print(f"    Logical Z count => {logicalZ_count_lld}")
        print(f"    Logical Y count => {logicalY_count_lld}")
        print(f" Logical Error rate => {logicalerror_count_lld / test_len}")
        
        print("-"*40)
        print()

        print("-------HLD Performance-------")
        print(f"           Total cases => {test_len_hld}")
        print(f"        Correct by HLD => {corrected_by_hld}")
        print(f"    Not correct by HLD => {test_len_hld - corrected_by_hld}")
        print(f" HLD ML model accuracy => {(corrected_by_hld/test_len_hld)*100} %")
        
        print()

        print(f"Logical Error count => {logicalerror_count_hld}")
        print(f"    Logical X count => {logicalX_count_hld}")
        print(f"    Logical Z count => {logicalZ_count_hld}")
        print(f"    Logical Y count => {logicalY_count_hld}")
        print(f" Logical Error rate => {logicalerror_count_hld / test_len}")
        
        print("-"*40)
        print()


    
    avg_logicalerror_lld = sum_logicalerror_count_lld / numinst
    avg_logicalX_lld = sum_logicalX_count_lld / numinst
    avg_logicalZ_lld = sum_logicalZ_count_lld / numinst
    avg_logicalY_lld = sum_logicalY_count_lld / numinst
    avg_acc_lld = sum_acc_count_lld / numinst

    avg_logicalerror_hld = sum_logicalerror_count_hld / numinst
    avg_logicalX_hld = sum_logicalX_count_hld / numinst
    avg_logicalZ_hld = sum_logicalZ_count_hld / numinst
    avg_logicalY_hld = sum_logicalY_count_hld / numinst
    avg_acc_hld = sum_acc_count_hld / numinst



    logicalerror_lld_list.append(avg_logicalerror_lld)
    logicalX_lld_list.append(avg_logicalX_lld)
    logicalZ_lld_list.append(avg_logicalZ_lld)
    logicalY_lld_list.append(avg_logicalY_lld)
    acc_lld_list.append(avg_acc_lld)

    logicalerror_hld_list.append(avg_logicalerror_hld)
    logicalX_hld_list.append(avg_logicalX_hld)
    logicalZ_hld_list.append(avg_logicalZ_hld)
    logicalY_hld_list.append(avg_logicalY_hld)
    acc_hld_list.append(avg_acc_hld)

    
    print("-"*40) 



sv = [prob_array, 
      logicalerror_lld_list, 
      logicalX_lld_list, 
      logicalZ_lld_list, 
      logicalY_lld_list, 
      acc_lld_list, 
      logicalerror_hld_list, 
      logicalX_hld_list, 
      logicalZ_hld_list, 
      logicalY_hld_list, 
      acc_hld_list]  

fileSaveName = "Accuracy_Error_" + str(prob_array[0]) + "-" + str(prob_array[-1]) + "_HLD_d=" + str(d) + ".csv"

np.savetxt(fileSaveName,sv,delimiter=",", fmt="%f") 





plt.figure(0)
plt.plot(prob_array, acc_lld_list, '.--', label='LLD accuracy')
plt.plot(prob_array, acc_hld_list, '.--', label='HLD accuracy')
plt.xlabel("Physical Error Probability")
plt.ylabel("Accuracy")
plt.legend()    
plt.title('Accuracy_d='+str(d))
plt.savefig('Accuracy_d='+str(d)+'.png') 




plt.figure(1)
xpoints = ypoints = plt.xlim()

plt.plot(prob_array, logicalerror_lld_list, '.-',label='depol,LLD')
plt.plot(prob_array, logicalX_lld_list, '.--',label='X,LLD')
plt.plot(prob_array, logicalZ_lld_list, '.--',label='Z,LLD')
plt.plot(prob_array, logicalY_lld_list, '.--',label='Y,LLD')

plt.plot(prob_array, logicalerror_hld_list, '.-',label='depol,HLD')
plt.plot(prob_array, logicalX_hld_list, '.--',label='X,HLD')
plt.plot(prob_array, logicalZ_hld_list, '.--',label='Z,HLD')
plt.plot(prob_array, logicalY_hld_list, '.--',label='Y,HLD')

plt.plot(xpoints, ypoints, color="cyan", label ='y=x')

axes = plt.gca()
axes.set_ylim([0,1])

axes.set_xlim([0,prob_array[-1]]) 
plt.xlabel("Physical Error Probability")
plt.ylabel("Logical Error Probability")
plt.legend()   
plt.title('Error_d='+str(d))
plt.savefig('Error_d='+str(d)+'.png')
