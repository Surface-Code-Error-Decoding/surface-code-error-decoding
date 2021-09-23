# High Level Decoder (HLD)

### main.py has to be executed and the variables shown below can be used to tweak data size, code distance, training hyperparameters. 
![image](https://user-images.githubusercontent.com/44721211/134567679-bc96de0c-faeb-4b4a-996d-be45dbde2ae8.png)
#### d => code distance
#### n => dataset size
#### numinst => number of execution instance for each physical error probability
#### split => Train test split ratio for ML model training and validation
#### epochs => training epochs
#### batch_size => batch size to train ML model

#### X,Y,Z probability can be tweaked accordingly depending on symmetric or asymmetric error is being used. 

---
### The code snippet shown below, that part of main.py creates a csv file. The data format in the file is self-explanatory from the code snippet. 
![image](https://user-images.githubusercontent.com/44721211/134567760-98b7d62c-42ab-4152-8885-7d60cfeb292c.png)

---
### The code snippet below creates two .png images saves to the current directory. First one is the accuracy plot, Second one is logical error plot. 
![image](https://user-images.githubusercontent.com/44721211/134567815-184e1f99-b036-46f7-9953-dddd9cc14f32.png)

---
