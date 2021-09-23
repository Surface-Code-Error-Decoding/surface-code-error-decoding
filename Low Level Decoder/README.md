# Low Level Decoder (LLD)

### main.py has to be executed and the variables shown below can be used to tweak data size, code distance, training hyperparameters. 
![image](https://user-images.githubusercontent.com/44721211/134566410-fb7c5357-caad-4e60-a563-432321efd4e9.png)
#### d => code distance
#### n => dataset size
#### numinst => number of execution instance for each physical error probability
#### split => Train test split ratio for ML model training and validation
#### epochs => training epochs
#### batch_size => batch size to train ML model

#### X,Y,Z probability can be tweaked accordingly depending on symmetric or asymmetric error is being used. 

---
### The code snippet shown below, that part of main.py creates a csv file. The data format in the file is self-explanatory from the code snippet. 
![image](https://user-images.githubusercontent.com/44721211/134566667-3f59adb9-3f3b-4615-b147-2d60ec0dc211.png)

---
### The code snippet below creates two .png images saves to the current directory. First one is the accuracy plot, Second one is logical error plot. 
![image](https://user-images.githubusercontent.com/44721211/134566942-e1662d45-b8d9-44c9-b980-2d4e53e8917a.png)

---
