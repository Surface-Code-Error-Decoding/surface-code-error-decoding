def mwpm_d3(data, d):

    X = data[:, 2*(d**2) : 2*(d**2)+((d+1)**2)].astype(float)
    y = data[:, 0 : 2*(d**2)]

    data_len = X.shape[0]
    y_pred = []
    
    ##MWPM implementation
    HZ = np.array([[0,0,0,1,0,0,1,0,0],
                    [1,1,0,1,1,0,0,0,0],
                    [0,0,0,0,1,1,0,1,1],
                    [0,0,1,0,0,1,0,0,0]])
    
    HX = np.array([[1,1,0,0,0,0,0,0,0],
                    [0,1,1,0,1,1,0,0,0],
                    [0,0,0,1,1,0,1,1,0],
                    [0,0,0,0,0,0,0,1,1]])
    

    MZ = Matching(HZ)
    MX = Matching(HX)

    for i in range(data_len):
        errorX = np.array([X[i][8], X[i][5], X[i][10], X[i][7]])
        errorZ = np.array([X[i][1], X[i][6], X[i][9], X[i][14]])

        correctionX = MZ.decode(errorX)
        correctionZ = MX.decode(errorZ)

        correction = np.array([correctionX, correctionZ])
        correction = np.transpose(correction)
        correction = correction.reshape(2*(d**2))

        y_pred.append(correction)

    
    y_pred = np.array(y_pred)


    print(f" Output Shape => {y_pred.shape}")

    ##predictions and accuracy
    acc = accuracy_score(y, y_pred)

    print("-"*40)
    print(f"MWPM Accuracy => {acc}")
    print("-"*40)
    
    return (X, y_pred, y)


def mwpm_d5(data, d):

    X = data[:, 2*(d**2) : 2*(d**2)+((d+1)**2)].astype(float)
    y = data[:, 0 : 2*(d**2)]

    data_len = X.shape[0]
    y_pred = []
    
    ##MWPM implementation
    HZ = np.array([[0,0,0,0,0,  1,0,0,0,0,  1,0,0,0,0,  0,0,0,0,0,  0,0,0,0,0],
                    [0,0,0,0,0,  0,0,0,0,0,  0,0,0,0,0,  1,0,0,0,0,  1,0,0,0,0],
                    [1,1,0,0,0,  1,1,0,0,0,  0,0,0,0,0,  0,0,0,0,0,  0,0,0,0,0],
                    [0,0,0,0,0,  0,0,0,0,0,  1,1,0,0,0,  1,1,0,0,0,  0,0,0,0,0],
                    
                    [0,0,0,0,0,  0,1,1,0,0,  0,1,1,0,0,  0,0,0,0,0,  0,0,0,0,0],
                    [0,0,0,0,0,  0,0,0,0,0,  0,0,0,0,0,  0,1,1,0,0,  0,1,1,0,0],
                    [0,0,1,1,0,  0,0,1,1,0,  0,0,0,0,0,  0,0,0,0,0,  0,0,0,0,0],
                    [0,0,0,0,0,  0,0,0,0,0,  0,0,1,1,0,  0,0,1,1,0,  0,0,0,0,0],
                    
                    [0,0,0,0,0,  0,0,0,1,1,  0,0,0,1,1,  0,0,0,0,0,  0,0,0,0,0],
                    [0,0,0,0,0,  0,0,0,0,0,  0,0,0,0,0,  0,0,0,1,1,  0,0,0,1,1],
                    [0,0,0,0,1,  0,0,0,0,1,  0,0,0,0,0,  0,0,0,0,0,  0,0,0,0,0],
                    [0,0,0,0,0,  0,0,0,0,0,  0,0,0,0,1,  0,0,0,0,1,  0,0,0,0,0]])
    

    HX = np.array([[1,1,0,0,0,  0,0,0,0,0,  0,0,0,0,0,  0,0,0,0,0,  0,0,0,0,0],
                    [0,0,1,1,0,  0,0,0,0,0,  0,0,0,0,0,  0,0,0,0,0,  0,0,0,0,0],
                    [0,1,1,0,0,  0,1,1,0,0,  0,0,0,0,0,  0,0,0,0,0,  0,0,0,0,0],
                    [0,0,0,1,1,  0,0,0,1,1,  0,0,0,0,0,  0,0,0,0,0,  0,0,0,0,0],
                    
                    [0,0,0,0,0,  1,1,0,0,0,  1,1,0,0,0,  0,0,0,0,0,  0,0,0,0,0],
                    [0,0,0,0,0,  0,0,1,1,0,  0,0,1,1,0,  0,0,0,0,0,  0,0,0,0,0],
                    [0,0,0,0,0,  0,0,0,0,0,  0,1,1,0,0,  0,1,1,0,0,  0,0,0,0,0],
                    [0,0,0,0,0,  0,0,0,0,0,  0,0,0,1,1,  0,0,0,1,1,  0,0,0,0,0],
                    
                    [0,0,0,0,0,  0,0,0,0,0,  0,0,0,0,0,  1,1,0,0,0,  1,1,0,0,0],
                    [0,0,0,0,0,  0,0,0,0,0,  0,0,0,0,0,  0,0,1,1,0,  0,0,1,1,0],
                    [0,0,0,0,0,  0,0,0,0,0,  0,0,0,0,0,  0,0,0,0,0,  0,1,1,0,0],
                    [0,0,0,0,0,  0,0,0,0,0,  0,0,0,0,0,  0,0,0,0,0,  0,0,0,1,1]])
    


    MZ = Matching(HZ)
    MX = Matching(HX)

    for i in range(data_len):

        errorX = np.array([X[i][12], X[i][24], 
                        X[i][7], X[i][19], 
                        X[i][14], X[i][26], 
                        X[i][9], X[i][21], 
                        X[i][16], X[i][28], 
                        X[i][11], X[i][23]])

        errorZ = np.array([X[i][1], X[i][3], 
                        X[i][8], X[i][10], 
                        X[i][13], X[i][15], 
                        X[i][20], X[i][22], 
                        X[i][25], X[i][27], 
                        X[i][32], X[i][34]])
        

        correctionX = MZ.decode(errorX)
        correctionZ = MX.decode(errorZ)

        correction = np.array([correctionX, correctionZ])
        correction = np.transpose(correction)
        correction = correction.reshape(2*(d**2))

        y_pred.append(correction)

    
    y_pred = np.array(y_pred)

    print(f" Output Shape => {y_pred.shape}")
    
    ##predictions and accuracy
    acc = accuracy_score(y, y_pred)

    print("-"*40)
    print(f"MWPM Accuracy => {acc}")
    print("-"*40)
    
    return (X, y_pred, y) 


    


def training_simple_FFNN_model(data, d, split=0.2, epochs=100, batch_size=1000):

    X = data[:, 2*(d**2) : 2*(d**2)+((d+1)**2)].astype(float)
    y = data[:, 0 : 2*(d**2)]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size = split, 
        random_state = 42, 
        shuffle = True
    )

    ##reshaping input data in square matrix form    
    train_len = X_train.shape[0]
    test_len = X_test.shape[0]


    ##FFNN model
    model_ffnn = Sequential()

    model_ffnn.add(Dense(64, input_dim = (d+1)**2, activation="relu"))
    model_ffnn.add(Dense(2*(d**2), activation="sigmoid"))

    # print(model_ffnn.summary())


    ##model compilation
    model_ffnn.compile(loss = "binary_crossentropy", 
                        optimizer = SGD(learning_rate=0.01), 
                        metrics = ["accuracy"])
    
    ##model training
    model_ffnn.fit(X_train, y_train, 
                    validation_split = 0.2,
                    epochs = epochs, 
                    batch_size = batch_size, 
                    verbose = 0) 
    
            
    ##predictions and accuracy 
    y_train_pred = model_ffnn.predict(X_train).round()
    train_acc = accuracy_score(y_train, y_train_pred)

    y_test_pred = model_ffnn.predict(X_test).round()
    test_acc = accuracy_score(y_test, y_test_pred)


    print("-"*40)
    print(f"Training Accuracy => {train_acc}")
    print(f"    Test Accuracy => {test_acc}")
    print("-"*40)
    
    return (X_test, y_test_pred, y_test)





def training_complex_FFNN_model(data, d, split=0.2, epochs=100, batch_size=1000):

    X = data[:, 2*(d**2) : 2*(d**2)+((d+1)**2)].astype(float)
    y = data[:, 0 : 2*(d**2)]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size = split, 
        random_state = 42, 
        shuffle = True
    )

    ##reshaping input data in square matrix form 
    train_len = X_train.shape[0]
    test_len = X_test.shape[0]


    ##FFNN model
    model_ffnn = Sequential()

    model_ffnn.add(Dense(64, input_dim = (d+1)**2, activation="relu"))
    model_ffnn.add(Dense(128, activation="relu"))
    model_ffnn.add(Dense(256, activation="relu"))
    model_ffnn.add(Dense(128, activation="relu"))
    model_ffnn.add(Dense(64, activation="relu"))
    model_ffnn.add(Dense(2*(d**2), activation="sigmoid"))

    # print(model_ffnn.summary())


    ##model compilation
    model_ffnn.compile(loss = "binary_crossentropy", 
                        optimizer = SGD(learning_rate=0.01), 
                        metrics = ["accuracy"])
    
    ##model training
    model_ffnn.fit(X_train, y_train, 
                    validation_split = 0.2,
                    epochs = epochs, 
                    batch_size = batch_size, 
                    verbose = 0)
    
            
    ##predictions and accuracy 
    y_train_pred = model_ffnn.predict(X_train).round()
    train_acc = accuracy_score(y_train, y_train_pred)


    y_test_pred = model_ffnn.predict(X_test).round()
    test_acc = accuracy_score(y_test, y_test_pred)


    print("-"*40)
    print(f"Training Accuracy => {train_acc}")
    print(f"    Test Accuracy => {test_acc}")
    print("-"*40)
    
    return (X_test, y_test_pred, y_test) 





def training_simple_CNN_model(data, d, split=0.2, epochs=100, batch_size=1000):

    X = data[:, 2*(d**2) : 2*(d**2)+((d+1)**2)].astype(float)
    y = data[:, 0 : 2*(d**2)]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size = split, 
        random_state = 42, 
        shuffle = True
    )

    ##reshaping input data in square matrix form 
    train_len = X_train.shape[0]
    test_len = X_test.shape[0]
    X_train = X_train.reshape(train_len, d+1, d+1, 1)
    X_test = X_test.reshape(test_len, d+1, d+1, 1)


    ##CNN model
    model_cnn = Sequential()
    model_cnn.add(Conv2D(64, kernel_size=2, activation="relu", input_shape=(d+1,d+1,1)))
    
    model_cnn.add(Flatten())
    model_cnn.add(Dense(256, activation="relu"))
    model_cnn.add(Dense(64, activation="relu"))
    model_cnn.add(Dense(2*(d**2), activation="sigmoid"))

    # print(model_cnn.summary())


    ##model compilation
    model_cnn.compile(loss = "binary_crossentropy", 
                        optimizer = SGD(learning_rate=0.01), 
                        metrics = ["accuracy"])
    
    ##model training
    model_cnn.fit(X_train, y_train, 
                    epochs = epochs, 
                    batch_size = batch_size, 
                    verbose = 0)
    
    
    ##predictions and accuracy 
    y_train_pred = model_cnn.predict(X_train).round()
    train_acc = accuracy_score(y_train, y_train_pred)

    y_test_pred = model_cnn.predict(X_test).round()
    test_acc = accuracy_score(y_test, y_test_pred)

    print("-"*40)
    print(f"Training Accuracy => {train_acc}")
    print(f"    Test Accuracy => {test_acc}")
    print("-"*40)
    
    ##flattening back each row of training data
    X_train = X_train.reshape(train_len, (d+1)**2)
    X_test = X_test.reshape(test_len, (d+1)**2)
    
    return (X_test, y_test_pred, y_test) 




def training_complex_CNN_model(data, d, split=0.2, epochs=100, batch_size=1000):

    X = data[:, 2*(d**2) : 2*(d**2)+((d+1)**2)].astype(float)
    y = data[:, 0 : 2*(d**2)]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size = split, 
        random_state = 42, 
        shuffle = True
    )

    ##reshaping input data in square matrix form 
    train_len = X_train.shape[0]
    test_len = X_test.shape[0]
    X_train = X_train.reshape(train_len, d+1, d+1, 1)
    X_test = X_test.reshape(test_len, d+1, d+1, 1)


    ##CNN model
    model_cnn = Sequential()
    model_cnn.add(Conv2D(64, kernel_size=2, activation="relu", input_shape=(d+1,d+1,1)))
    model_cnn.add(Conv2D(64, kernel_size=2, activation="relu", input_shape=(d+1,d+1,1)))
    model_cnn.add(Conv2D(64, kernel_size=2, activation="relu", input_shape=(d+1,d+1,1)))
    
    
    model_cnn.add(Flatten())
    model_cnn.add(Dense(512, activation="relu"))
    model_cnn.add(Dense(256, activation="relu"))
    model_cnn.add(Dense(128, activation="relu"))
    model_cnn.add(Dense(64, activation="relu"))
    model_cnn.add(Dense(2*(d**2), activation="sigmoid"))

    # print(model_cnn.summary())


    ##model compilation
    model_cnn.compile(loss = "binary_crossentropy", 
                        optimizer = SGD(learning_rate=0.01), 
                        metrics = ["accuracy"])
    
    ##model training
    model_cnn.fit(X_train, y_train, 
                    epochs = epochs, 
                    batch_size = batch_size, 
                    verbose = 0)
    
    
    ##predictions and accuracy 
    y_train_pred = model_cnn.predict(X_train).round()
    train_acc = accuracy_score(y_train, y_train_pred)

    y_test_pred = model_cnn.predict(X_test).round()
    test_acc = accuracy_score(y_test, y_test_pred)

    print("-"*40)
    print(f"Training Accuracy => {train_acc}")
    print(f"    Test Accuracy => {test_acc}")
    print("-"*40)
    
    ##flattening back each row of training data
    X_train = X_train.reshape(train_len, (d+1)**2)
    X_test = X_test.reshape(test_len, (d+1)**2)
    
    return (X_test, y_test_pred, y_test)