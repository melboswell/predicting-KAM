
# coding: utf-8

# In[17]:


import scipy.io as sio
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
import keras
from keras import regularizers
from keras.models import Sequential, Model
from keras.initializers import glorot_normal
from keras.layers import Dense, LSTM, TimeDistributed, Activation, Dropout, Input, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape, Flatten
from keras.optimizers import Adam
import pickle
import keras.backend as K
from keras.losses import mean_squared_error
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import mpld3
mpld3.enable_notebook()

# add r^2 as outcome metric
from keras import backend as K
def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def r2_numpy(data,labels,model):
    y_pred2 = model.predict(data)
    eps = np.finfo(float).eps

    SS_res =  np.sum(np.square(labels - y_pred2)) 
    SS_tot = np.sum(np.square(labels - np.mean(labels))) 
    r2 = ( 1 - SS_res/(SS_tot + eps) )
    mse = np.mean(np.sqrt(np.square(y_pred2-labels)))
    return r2, mse


# Prepare the data

# In[2]:


config = tf.ConfigProto(device_count = {'CPU' : 1, 'GPU' : 0})
session = tf.Session(config=config)
K.set_session(session)


# In[3]:


# Import data
markersIn = sio.loadmat('../FullyConnected/input.mat')
KAM = sio.loadmat('../FullyConnected/KAM.mat')
subind = sio.loadmat('../FullyConnected/subind.mat')

# Define input (markers) and output (KAM)
markersIn = markersIn['markers']
KAMIn = KAM['KAM']
subind = subind['subind']
subind = subind -1 # Change to python indexing


# In[5]:


# Adjust input to correct format
markers = np.moveaxis(markersIn, [0,1,2], [2,1,0])
print('Original markers.shape=' + str(markers.shape))

markersOnly = markers[:,:,0:39]
print('markersOnly.shape=' + str(markersOnly.shape))

height = markers[:,:,39] 
weight = markers[:,:,40] 
legBin = markers[:,:,41] 
print('height shape = ' + str(height.shape))

# Remove Weight from markers
inds = [] ;

inds2 = range(0,30,1) # range of timesteps to take
inds3 = np.concatenate((range(0,39),[41]))
markers = markers[:,:,inds3]
markersIn = markers[:,inds2,:]
print('New Markers Shape ' + str(markersIn.shape))

# print(markersIn[1,1,:])


# In[6]:


# Adjust output to correct format
KAM = KAMIn.reshape((-1,30,1))
# Normalize KAM data by height and weight
KAM = np.divide(KAM,weight.reshape(-1,30,1))
KAM = np.divide(KAM,height.reshape(-1,30,1))*100
print(KAM.shape)

# Normalize marker data by height
markers = np.divide(markersIn,height[:,inds2].reshape(-1,markersIn.shape[1],1))

KAMP1 = np.max(KAM[:,range(15),:],axis=1)
KAMP1 = KAMP1.reshape((-1,1,1))
print(KAMP1.shape)


# In[7]:


# Divide into Test, Dev, Train
np.random.seed(0)
maxVal = np.max(subind)
subShuff = np.arange(0,maxVal+1)
np.random.shuffle(subShuff)
trainNums = subShuff[0:80]
devNums = subShuff[80:90]
testNums = subShuff[90:98]

labelVar = KAMP1

trainInds = np.array(0)
for i in trainNums:
    trainInds = np.append(trainInds,np.argwhere(subind==i)[:,1])
trainInds = trainInds[1:]
    
devInds = np.array(0)
for i in devNums:
    devInds = np.append(devInds,np.argwhere(subind==i)[:,1])
devInds = devInds[1:]

testInds = np.array(0)
for i in testNums:
    testInds = np.append(testInds,np.argwhere(subind==i)[:,1])
testInds = testInds[1:]

# Build training dataset and labels # Doing Peaks Now
inputTrain = markers[trainInds,:,:]
labelTrain = labelVar[trainInds,:,:]
# inputTrain = inputTrain.reshape((inputTrain.shape[0],-1))

inputDev = markers[devInds,:,:]
labelDev = labelVar[devInds,:,:]
# inputDev = inputDev.reshape((inputDev.shape[0],-1))

inputTest = markers[testInds,:,:]
labelTest = labelVar[testInds,:,:]
# inputTest = inputTest.reshape((inputTest.shape[0],-1))

print('inputTrain size = ' + str(inputTrain.shape))
print('labelTrain size = ' + str(labelTrain.shape))


# In[9]:


# Flatten Input and Output Data
mTrain = inputTrain.shape[0] # number of examples to train on 
# mTrain =1000 # number of examples to train on 

#Input is mx328
trainData = inputTrain[0:mTrain,:]
devData = inputDev
testData = inputTest

print("trainData shape = " + str(trainData.shape))

# LabelsFull = np.reshape(labelTrain,(labelTrain.shape[0]*labelTrain.shape[1],labelTrain.shape[2]))
# trainLabels = LabelsFull[0:mTrain,:]
# devLabelsFull = np.reshape(labelDev,(labelDev.shape[0]*labelDev.shape[1],labelDev.shape[2]))
# devLabels = devLabelsFull
# testLabelsFull = np.reshape(labelTest,(labelTest.shape[0]*labelTest.shape[1],labelTest.shape[2]))
# testLabels = testLabelsFull

LabelsFull = np.reshape(labelTrain,(labelTrain.shape[0],labelTrain.shape[1],labelTrain.shape[2]))
trainLabels = LabelsFull[0:mTrain,:]
trainLabels = np.squeeze(np.transpose(trainLabels,[0,2,1]),axis=2)
devLabelsFull = np.reshape(labelDev,(labelDev.shape[0],labelDev.shape[1],labelDev.shape[2]))
devLabels = np.squeeze(np.transpose(devLabelsFull,[0,2,1]),axis=2)
testLabelsFull = np.reshape(labelTest,(labelTest.shape[0],labelTest.shape[1],labelTest.shape[2]))
testLabels = np.squeeze(np.transpose(testLabelsFull,[0,2,1]),axis=2)

Tx = trainData.shape[1]
nx = trainData.shape[2]

print("trainLabels shape = " + str(trainLabels.shape))
print("Tx = " + str(Tx) + ', nx = ' + str(nx))


# Construct Model

# In[10]:


def make_model(input_shape):
    """
    Function creating the model's graph in Keras.
    
    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)

    Returns:
    model -- Keras model instance
    """
    
    X_input = Input(shape = input_shape)
    
    X = BatchNormalization(input_shape=input_shape)(X_input)
    # Step 1: CONV layer (≈4 lines)
    X = Conv1D(20,input_shape=input_shape,kernel_size=5,strides=1,activation='relu',padding='same',use_bias = True)(X_input)  # CONV1D
    X = BatchNormalization()(X)                                 # Batch normalization
    X = Conv1D(40,kernel_size=5,strides=1,activation='relu',padding='same',use_bias = True)(X)  # CONV1D
    X = BatchNormalization()(X) 
    X = Conv1D(60,kernel_size=5,strides=1,activation='relu',padding='valid',use_bias = True)(X)  # CONV1D
    X = BatchNormalization()(X) 
    X = Conv1D(80,kernel_size=5,strides=1,activation='relu',padding='valid',use_bias = True)(X)  # CONV1D
    
    # Step 2: Dense Layers
    X = Flatten()(X)
    X = Dense(150, activation = 'relu')(X)
    X = Dense(75,activation = 'relu')(X)
    X = Dense(1,activation = 'linear')(X)
#     X = TimeDistributed(Dense(1, activation = "linear"))(X) 

    model = Model(inputs = X_input, outputs = X)
    
    return model  

model = make_model(input_shape = (Tx,nx))

opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='mean_squared_error', optimizer=opt,metrics=[r2_keras])


# Run Model

# In[11]:


model.summary()


# In[14]:


nEpochs = 40
train_r2 = np.zeros((nEpochs,1))
dev_r2 = np.zeros((nEpochs,1))
train_loss = np.zeros((nEpochs,1))
dev_loss = np.zeros((nEpochs,1))

for i in range(nEpochs):
    print('Epoch ' + str(i+1) + ' of ' + str(nEpochs) + '.')

    history = model.fit(trainData,trainLabels, epochs=1 , batch_size = 32, shuffle = True, verbose=2)
    
    train_r2[i] = history.history['r2_keras']
    train_loss[i] = history.history['loss']
    dev_r2[i], dev_loss[i] = r2_numpy(devData,devLabels,model)    
    print('Train_loss = ' + str(train_loss[i]) + ', Train_r2 = ' + str(train_r2[i]) + ', Dev_loss = ' + str(dev_loss[i]) + ', Dev_r2 = ' + str(dev_r2[i]))

model.save('CNN_peakpred.hdf5')


# In[15]:


metrics = [train_r2,train_loss,dev_r2,dev_loss,epochCount]

with open('CNN_peakPrediction.pickle','wb') as f:
    pickle.dump([model, history, metrics], f)


# In[12]:


with open('CNN_peakPrediction.pickle', 'rb') as f:
    model, history, metrics = pickle.load(f)
    
train_r2,train_loss,dev_r2,dev_loss,epochCount =metrics


# Evaluate Performance

# In[22]:


lossPlt = plt.plot(train_loss)
DevlossPlt = plt.plot(dev_loss)

plt.ylabel('Mean Squared Error')
plt.xlabel('Epoch Number');
plt.legend(('Training','Dev'))

plt.figure(2)
r2Plt = plt.plot(train_r2)
devr2Plt = plt.plot(dev_r2)
plt.ylim([0, 1])
plt.ylabel('r^2')
plt.xlabel('Epoch Number');
plt.legend(('Training','Dev'))


# In[111]:


# Predict on some dev Data
visualizeRange = range(800,810)
data = trainData;
labels = trainLabels;

y_pred = np.reshape(model.predict(data[visualizeRange,:,:]),(-1,1))

print(y_pred.shape)

truePlot = plt.plot(np.reshape(np.squeeze(labels[visualizeRange,:]),(-1,1)))
predPlot = plt.plot(np.squeeze(y_pred).T)
testData.shape

plt.ylabel('KAM Peak Comparison')
plt.xlabel('Time')
plt.legend(('True','Predicted'),loc=4);

plt.savefig('KAM comparisons.png')


# In[20]:


model = keras.models.load_model('CNN_peakpred.hdf5',custom_objects = {'r2_keras':r2_keras})


# In[196]:


print(trainData.shape)
print(trainLabels.shape)


# In[22]:


r2_train, mse_train = r2_numpy(trainData,trainLabels,model)
r2_dev, mse_dev = r2_numpy(devData,devLabels,model)
r2_test, mse_test = r2_numpy(testData,testLabels,model)

print('Train r^2, then rmse = ' + str([r2_train, np.sqrt(mse_train)]))
print('Dev r^2, then rmse = ' + str([r2_dev, np.sqrt(mse_dev)]))
print('Test r^2, then rmse = ' + str([r2_test, np.sqrt(mse_test)]))

