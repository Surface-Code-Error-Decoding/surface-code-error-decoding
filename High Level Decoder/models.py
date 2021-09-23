import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import LabelEncoder 

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import SGD
from keras.utils import np_utils


def training_FFNN_model_LLD(X_train, y_train_lld, d, epochs=100, batch_size=1000):

    train_len = X_train.shape[0]

    ##FFNN model
    model_lld = Sequential()

    model_lld.add(Dense(64, input_dim = (d+1)**2, activation="relu"))
    model_lld.add(Dense(2*(d**2), activation="sigmoid"))

    # print(model_ffnn.summary())


    ##model compilation
    model_lld.compile(loss = "binary_crossentropy", 
                        optimizer = SGD(learning_rate=0.01), 
                        metrics = ["accuracy"])
    
    ##model training
    model_lld.fit(X_train, y_train_lld, 
                    validation_split = 0.2, 
                    epochs = epochs, 
                    batch_size = batch_size, 
                    verbose = 0)
    
            
    ##predictions and accuracy 
    y_train_pred = model_lld.predict(X_train).round()
    train_acc = accuracy_score(y_train_lld, y_train_pred)



    print("-"*40)
    print()
    print(f"LLD Training Accuracy => {train_acc}")
    
    
    
    return model_lld 



def training_FFNN_model_HLD(X_train, y_train_hld, d, epochs=100, batch_size=1000):

    train_len = X_train.shape[0]

    encoder = LabelEncoder()
    encoder.fit(y_train_hld)
    encoded_y = encoder.transform(y_train_hld) 

    dummy_y = np_utils.to_categorical(encoded_y) 


    ##FFNN model
    model_hld = Sequential()

    model_hld.add(Dense(64, input_dim = (d+1)**2, activation="relu"))
    model_hld.add(Dense(32, activation="relu"))
    model_hld.add(Dense(4, activation="softmax"))

    # print(model_ffnn.summary())


    ##model compilation
    model_hld.compile(loss = "categorical_crossentropy", 
                        optimizer = SGD(learning_rate=0.01), 
                        metrics = ["accuracy"])
    
    ##model training
    model_hld.fit(X_train, dummy_y, 
                    validation_split = 0.2, 
                    epochs = epochs, 
                    batch_size = batch_size, 
                    verbose = 0)
    
            
    ##predictions and accuracy 
    y_train_pred = model_hld.predict(X_train)
    y_train_pred = np.array([np.argmax(i) for i in y_train_pred])
    train_acc = accuracy_score(y_train_hld, y_train_pred)



    print("-"*40)
    print()
    print(f"HLD Training Accuracy => {train_acc}")
    
    
    
    return model_hld 

