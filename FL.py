# -*- coding: utf-8 -*-


import os
from google.colab import drive

MOUNTPOINT = "/content/drive"

DATADIR = os.path.join(MOUNTPOINT, "MyDrive")
drive.mount(MOUNTPOINT)

import numpy as np
from numpy import unique
from collections import Counter
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow import keras
#import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from keras.utils import np_utils
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, Conv1D
from tensorflow.keras.layers import MaxPool2D, MaxPooling1D
from keras import backend as K
from keras import backend
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import matthews_corrcoef

import time
import os
import psutil
import csv
from itertools import repeat
from PIL import Image
from numpy import asarray
import matplotlib.pyplot as plt

"""# Load the data"""

import pandas as pd
import io
 
path = "/content/drive/MyDrive/Federated Learning/Dataset/xerces1.4.1.csv"
data = pd.read_csv(path)
#data = pd.read_csv(io.BytesIO(uploaded['ant1.7.1.csv']))
#data = pd.read_csv(Mydrive/FederatedLearning/ant1.7.1.csv)
print(data)

"""# Data with dependent and independent variables only"""

data.drop(data.columns[[0, 1, 2, 3]], axis=1, inplace=True)
data.head()

"""# Listing all the columns with number of NULL values"""

data.isnull().sum()

"""# Columns with NULL values
- nr and nml are the columns with NULL values

# Replacing NULL values using 
- Mean Imputation
- Median Imputation
"""

data['nr'] = data['nr'].fillna(data['nr'].mean())
data['nml'] = data['nml'].fillna(data['nml'].median())

data.head()

"""# Listing all the columns with number of NULL values"""

data.isnull().sum()

"""### All the null values are replaced with the mean values of the respective columns"""

data.dtypes

"""- In case the data is not numeric convert using the following code"""

#data["nr"] = pd.to_numeric(data["nr"])
#data["nml"] = pd.to_numeric(data["nml"])

#data.dtypes
#data.fillna(data.mean())

#data.dtypes

data.shape

data.dtypes

dataset = data.values
dataset

#dataset.shape()

X_full = data.drop(['bugs'], axis = 1)
X_full.head()

Y_full = data['bugs']

X_full.head()

Y_full


#X_full = X_full.reshape(X_full.shape[0], X_full.shape[1], 1)
print(X_full.shape) 
#print((X_full.shape[0], X_full.shape[1], 1))
#print(unique(Y_full))

"""# Number of Bugs data points"""

print(data['bugs'].value_counts())

"""# Train Test Split"""

xTrain, xTest, yTrain, yTest = train_test_split(X_full, Y_full, test_size=0.20)
print(xTrain)
print(yTrain)

#scaler = StandardScaler()
#xTrain = scaler.fit_transform(xTrain)
#xTest = scaler.transform(xTest)
print(xTrain)
print(xTest)

"""# Oversampling """

print("Before oversampling: ",Counter(yTrain))
print("Before OverSampling, counts of label '1': {}".format(sum(yTrain == 1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(yTrain == 0)))

from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state = 0)
xTrain_res, yTrain_res = sm.fit_resample(xTrain, yTrain.ravel())
  
print('After OverSampling, the shape of train_X: {}'.format(xTrain_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(yTrain_res.shape))
  
print("After OverSampling, counts of label '1': {}".format(sum(yTrain_res == 1)))
print("After OverSampling, counts of label '0': {}".format(sum(yTrain_res == 0)))

xTrain=xTrain_res
print(xTrain)

yTrain=yTrain_res
print(yTrain)


"""# Reshape Features"""

#xTrain = xTrain.reshape(xTrain.shape[0], xTrain.shape[1], 1)
print(xTrain.shape)

#xTest = xTest.reshape(xTest.shape[0], xTest.shape[1], 1)
print(xTest.shape)

print(yTrain.shape)

print(yTest.shape)

#print((X_full.shape[0], X_full.shape[1], 1))
#print(unique(Y_full))

xTrain.shape[1]



"""# Define Model"""

algoName='CNN' #CNN, ANN, DNN

# xTrain = xTrain.astype('float32')
# xTest = xTest.astype('float32')

# xTrain = xTrain / 255.
# xTest = xTest / 255.

#if(algoName=='CNN'):
    #xTrain = np.expand_dims(xTrain, axis=2)
    #xTest = np.expand_dims(xTest, axis=2)

outputClasses=len(set(Y_full))
outputClasses

"""# One hot encoding"""

#yTrain = np.array(to_categorical(yTrain))
#yTest = np.array(to_categorical(yTest))
#Y_full = np.array(to_categorical(Y_full))

print("xTrain", xTrain.shape)
print("yTrain", yTrain.shape)
print()
print("xTest", xTest.shape)
print("yTest", yTest.shape)

yTrain

yTest

Y_full

"""# FOR CNN, DATASET RESHAPING"""

img_rows, img_cols = 1, xTrain.shape[1]
xTrain1 = xTrain
yTrain1 = yTrain
xTest1 = xTest
yTest1 = yTest
xTrain = xTrain1.values.reshape(xTrain1.shape[0], img_rows, img_cols, 1)
xTest = xTest.values.reshape(xTest1.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

"""# FOR TEST SPLIT"""

xServer, xClients, yServer, yClients = train_test_split(xTrain, yTrain, test_size=0.20) 

print("xServer", xServer.shape)
print("yServer", yServer.shape)
print()
print("xClients", xClients.shape)
print("yClients", yClients.shape)
print()
print("X_full", X_full.shape)
print("Y_full", Y_full.shape)

def my_metrics(y_true, y_pred):
    accuracy=accuracy_score(y_true, y_pred)
    precision=precision_score(y_true, y_pred,average='weighted', zero_division=1)
    recall=recall_score(y_true, y_pred, average='weighted')
    f1Score=f1_score(y_true, y_pred, average='weighted') 
    fpr_keras, tpr_keras, thresholds = roc_curve(y_true, y_pred)
    auc_keras = auc(fpr_keras, tpr_keras)
    mcc = matthews_corrcoef(y_true, y_pred)
    #AUC=metrics.roc_auc_score(y_true, y_pred)
    print("Accuracy  : {}".format(accuracy))
    print("Precision : {}".format(precision))
    print("Recall : {}".format(recall))
    print("f1Score : {}".format(f1Score))
    print("AUC : {}".format(auc_keras))
    print("MCC : {}".format(mcc))
    cm=confusion_matrix(y_true, y_pred)
    print(cm)
    return accuracy, precision, recall, f1Score, auc_keras, mcc

verbose = 0
epochs = 100
batch_size = 10
activationFun='relu'
optimizerName='adam'

def createDeepModel():
    model = Sequential()
    
    if(algoName=='CNN'):     
        model.add(Conv2D(filters=100, kernel_size=1, activation=activationFun,input_shape = input_shape))
        #model.add(MaxPool2D(pool_size=(1,8)))
        model.add(Conv2D(filters=150, kernel_size=1, activation=activationFun))
        #model.add(MaxPool2D(pool_size=(1,8)))
        #model.add(Conv2D(filters=256, kernel_size=1, activation=activationFun))
        #model.add(MaxPool2D(pool_size=(1,8)))
        model.add(Conv2D(filters=128, kernel_size=1, activation=activationFun))
        #model.add(MaxPool2D(pool_size=(1,8)))
        #model.add(Dropout(0.05))
        #model.add(MaxPool2D(pool_size=(1,8)))
        #model.add(BatchNormalization())
        
        model.add(Flatten())
        model.add(Dense(1024, activation=activationFun))
        model.add(Dropout(0.06))
        #model.add(BatchNormalization())
        model.add(Dense(512, activation=activationFun))
        model.add(Dropout(0.05))
        #model.add(BatchNormalization())
        model.add(Dense(128, activation=activationFun))
        model.add(Dropout(0.05))
        model.add(Dense(64, activation=activationFun))
        model.add(Dropout(0.02))
        model.add(Dense(16, activation=activationFun))
        #model.add(Dropout(0.5))
        #model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=optimizerName, metrics=['accuracy'])
        
    elif(algoName=='ANN'):
        #model.add(Dense(15, activation = 'relu', input_dim=xTrain.shape[1]))
        #model.add(Dense( 8,  activation = 'relu'))
        #model.add(Dense(5,  activation = 'relu'))
        #model.add(Dense(1,  activation = 'sigmoid'))
        model.add(Dense(128, input_dim=xTrain.shape[1], activation=activationFun))
        #model.add(Flatten())
        model.add(Dropout(0.05))
        #model.add(Dense(200, activation=activationFun))
        #model.add(Dropout(0.05))
        model.add(Dense(64, activation=activationFun))
        model.add(Dropout(0.05))
        model.add(Dense(32, activation=activationFun))
        #model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=optimizerName, metrics=['accuracy'])
        
    elif(algoName=='DNN'):
        model.add(Dense(128, input_dim=xTrain.shape[1], activation=activationFun))
        #model.add(Flatten())
        model.add(Dropout(0.05))
        model.add(Dense(128, activation=activationFun))
        #model.add(Dropout(0.05))
        model.add(Dense(64, activation=activationFun))
        model.add(Dropout(0.05))
        model.add(Dense(64, activation=activationFun))
        model.add(Dropout(0.05))
        model.add(Dense(32, activation=activationFun))
        model.add(Dropout(0.05))
        model.add(Dense(32, activation=activationFun))
        model.add(Dense(24, activation=activationFun))
        model.add(Dropout(0.05))
        model.add(Dense(16, activation=activationFun))
        model.add(Dropout(0.05))
        model.add(Dense(8, activation=activationFun))
        model.add(Dropout(0.05))
        model.add(Dense(5,  activation=activationFun))
        #model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=optimizerName, metrics=['accuracy'])
    return model

def predictTestData(yPredict, yTest):
    #Converting predictions to label
    print("yPredict",len(yPredict))
    #pred = list()
    #for i in range(len(yPredict)):
        #pred.append(np.argmax(yPredict[i]))
    #print(yPredict)    
    
    #Converting one hot encoded test label to label
    #test = list()
    #for i in range(len(yTest)):
        #test.append(np.argmax(yTest[i]))
    #print(test)
    return my_metrics(yTest, yPredict)

def sumOfWeights(weights):
    return sum(map(sum, weights))

def getWeights(model):
    allLayersWeights=deepModel.get_weights()
    return allLayersWeights
    
# Initially train central deep model
deepModel = createDeepModel()

numOfIterations=20

numOfClients=10 # 10, 15, 20, 25, 30, 35, 40, 45, 50

#modelLocation=root_path+ "Ant1.7" +"Models/" +str(algoName) +"_Sync_users_" +str(numOfClients) +"_" +activationFun +"_" +optimizerName +"_FL_Model.h5"
    
accList, precList, recallList, f1List, aucList, mccList= [], [], [], [], [], []

deepModelAggWeights=[]

firstClientFlag=True

def updateServerModel(clientModel, clientModelWeight):
    global firstClientFlag
    for ind in range(len(clientModelWeight)):
        if(firstClientFlag==True):
            deepModelAggWeights.append(clientModelWeight[ind])            
        else:
            deepModelAggWeights[ind]=(deepModelAggWeights[ind]+clientModelWeight[ind])

def updateClientsModels():
    global clientsModelList
    global deepModel
    clientsModelList.clear()
    for clientID in range(numOfClients):
        m = keras.models.clone_model(deepModel)
        m.set_weights(deepModel.get_weights())
        clientsModelList.append(m)

# ----- 1. Train central model initially -----
def trainInServer():
    deepModel.fit(xServer, yServer, epochs=epochs, batch_size=batch_size, verbose=verbose)
    #deepModel.fit(X_full, Y_full, epochs=epochs, batch_size=batch_size, verbose=verbose)
    deepModel.save('/content/drive/MyDrive/Federated Learning/Model/xerces1.4.4.1CNNModel.h5')
trainInServer()

# ------- 2. Separate clients data into lists ----------
xClientsList=[]
yClientsList=[]
clientsModelList=[]
clientDataInterval=len(xClients)//numOfClients
lastLowerBound=0

for clientID in range(numOfClients):
    xClientsList.append(xClients[lastLowerBound : lastLowerBound+clientDataInterval])
    yClientsList.append(yClients[lastLowerBound : lastLowerBound+clientDataInterval])
    model=load_model('/content/drive/MyDrive/Federated Learning/Model/xerces1.4.4.1CNNModel.h5')
    clientsModelList.append(model)
    lastLowerBound+=clientDataInterval

# ------- 3. Update clients' model with intial server's deep-model ----------
for clientID in range(numOfClients):
    clientsModelList[clientID].fit(xClientsList[clientID], yClientsList[clientID], epochs=epochs, batch_size=batch_size, verbose=verbose)
        
start_time = time.time()
process = psutil.Process(os.getpid())
for iterationNo in range(1,numOfIterations+1):
    print("Iteration",iterationNo)
    for clientID in range(numOfClients):
        print("clientID",clientID)
        clientsModelList[clientID].compile(loss='binary_crossentropy', optimizer=optimizerName, metrics=['accuracy'])
        clientsModelList[clientID].fit(xClientsList[clientID], yClientsList[clientID], epochs=epochs, batch_size=batch_size, verbose=verbose)
        clientWeight=clientsModelList[clientID].get_weights()
        # Find sum of all client's model
        updateServerModel(clientsModelList[clientID], clientWeight)
        firstClientFlag=False
    #Avarage all clients model
    for ind in range(len(deepModelAggWeights)):
        deepModelAggWeights[ind]/=numOfClients

    dw_last=deepModel.get_weights()

    for ind in range(len(deepModelAggWeights)): 
        dw_last[ind]=deepModelAggWeights[ind]
     
    #Update server's model
    deepModel.set_weights(dw_last) 
    print("Server's model updated")
    print("Saving model . . .")
    deepModel.save('/content/drive/MyDrive/Federated Learning/Model/xerces1.4.4.1CNNModel.h5')
    # Servers model is updated, now it can be used again by the clients
    updateClientsModels()
    firstClientFlag=True
    deepModelAggWeights.clear()

    yPredict = deepModel.predict(xTest)
    #print(yPredict)
    acc, prec, recall, f1Score, aucScore, mccScore= predictTestData(yPredict.round(), yTest)
    
    #AUC = metrics.roc_auc_score(yPredict, yTest)
    accList.append(acc)
    precList.append(prec)
    recallList.append(recall)
    f1List.append(f1Score)
    aucList.append(aucScore)
    mccList.append(mccScore)
    print("Acc:\n", acc)
    print("Prec:\n", prec)
    print("Recall:\n", recall)
    print("F1-Score:\n", f1Score)
    print("AUC value is:\n", aucScore)
    print("MCC value is:\n", mccScore)

memoryTraining=process.memory_percent()
timeTraining=time.time() - start_time
print("---Memory---",memoryTraining)
print("--- %s seconds (TRAINING)---" % (timeTraining))

early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')

history = deepModel.fit(xServer, yServer, epochs=epochs, 
                        validation_data = (xTest,yTest))
                        # callbacks=[early_stopping])

learningAccs=history.history['val_accuracy']
learningLoss=history.history['val_loss']

# resultSaveLocation=root_path+'Results/'+algoName+'_Users_vs_TR_vs_Iterations_vs_AccLossMemTime'+'.csv'
dfSave=pd.DataFrame(columns=['Clients', 'Iterations to converge', 'Accuracy', 'Loss', 'Memory', 'Time'])
dfSaveIndex=0
saveList = [numOfClients, len(learningLoss), learningAccs[len(learningAccs)-1], learningLoss[len(learningLoss)-1], memoryTraining, timeTraining]
dfSave.loc[dfSaveIndex] = saveList

yPredict = deepModel.predict(xTest)
acc, prec, recall, f1Score, aucScore, mccScore= predictTestData(yPredict.round(), yTest)

#AUC = metrics.roc_auc_score(y_test, yPredict)

print("Number of users:", numOfClients)
deepModel.save('/content/drive/MyDrive/Federated Learning/Model/xerces1.4.4.1CNNModel.h5')
print("Epochs:", epochs)
print("BatchSize:", batch_size)
print("Activation:", activationFun, "Optimizer:", optimizerName)

print("Iterations:", numOfIterations)
print("Memory:", memoryTraining)
print("Time:", timeTraining)
print(dfSave)

df_performance_timeRounds = pd.DataFrame(
    {'Accuracy': accList,
     'Precision': precList,
     'Recall': recallList,
     'F1-Score': f1List,
     'AUC': aucList,
     'MCC': mccList
    })

df_performance_timeRounds



