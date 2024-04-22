# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 14:46:51 2023

@author: riyad
"""

class Algorithms:
    def __init__(self, *args, **kwargs):
        pass
    
    def fitSVM(trainingData, trainingLabel):
        # from sklearn.pipeline import make_pipeline
        # from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        
        SVM =  SVC(random_state=9, C=4, degree=3, gamma=200,kernel='rbf')
        trainedSVM = SVM.fit(trainingData, trainingLabel)
        
        return trainedSVM
    
    def fitRF(trainData, trainLabel):
        from sklearn.ensemble import RandomForestClassifier
        import numpy as np
        
        RFClassifier = RandomForestClassifier(random_state=9, criterion ='entropy', n_estimators = 100)
        trainedRFClassifier = RFClassifier.fit(trainData, trainLabel)
        return trainedRFClassifier
    
    def fitANN(trainData_ANN, trainLabel_ANN, testData_ANN, testLabel_ANN):        
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, Activation
        from tensorflow.keras.optimizers import Adam
        
        from tensorflow.keras.callbacks import Callback
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
        
        import time
        from datetime import datetime
        import tensorboard
        import numpy as np
        
        np.random.seed(9)  # set random seed to a fixed value for reproducibility
        
        model = Sequential()
        model.add(Dense(100, activation="relu", input_dim=trainData_ANN.shape[1]))
        model.add(Dropout(0.2))
        model.add(Dense(100, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(80, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(70, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(60, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(50, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(40, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(30, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(20, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(len(trainLabel_ANN[0]), activation="softmax"))
    
        adam = Adam(learning_rate=0.001)
        model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy", "mae"]) 
        
        
        
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1)
        
        class timecallback(tf.keras.callbacks.Callback):
            time.clock = time.time
            def __init__(self):
                self.times = []
                # use this value as reference to calculate cummulative time taken
                self.timetaken = time.time()
            def on_epoch_end(self,epoch,logs = {}):
                self.times.append((epoch,time.time() - self.timetaken))
            def on_train_end(self,logs = {}):
                # plt.xlabel('Epoch')
                # plt.ylabel('Total time taken until an epoch in seconds')
                # plt.plot(*zip(*self.times))
                # plt.show()
                print(self.times)
                return self.times
                
        # Create the callback instance
        time_callback = timecallback()
        
        historyANN_NoIndicesModel= model.fit(trainData_ANN, trainLabel_ANN, 
                                                 epochs=500, batch_size=16, verbose=2,  
                                                 validation_data=(testData_ANN, testLabel_ANN),
                                                 callbacks=[time_callback, early_stopping])
        
        print('I am done')
        return historyANN_NoIndicesModel
    
        