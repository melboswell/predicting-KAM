
# coding: utf-8

# In[2]:


import scipy.io as sio
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
import keras
from keras import regularizers
from keras.models import Sequential
from keras.initializers import glorot_normal
from keras.layers import Dense, LSTM, TimeDistributed, BatchNormalization, Input, Dropout, Flatten,Bidirectional
import keras.backend as K
from keras.losses import mean_squared_error
import pickle
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

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

# In[3]:


config = tf.ConfigProto(device_count = {'CPU' : 1, 'GPU' :1})
session = tf.Session(config=config)
K.set_session(session)


# In[4]:


# Import data
markersIn = sio.loadmat('input.mat')
KAM = sio.loadmat('KAM.mat')
subind = sio.loadmat('subind.mat')

# Define input (markers) and output (KAM)
markersInput = markersIn['markers']
KAMIn = KAM['KAM']
subind = subind['subind']
subind = subind -1 # Change to python indexing


# In[5]:


# Adjust input to correct format
markers = np.moveaxis(markersInput, [0,1,2], [2,1,0])
print('Original markers.shape=' + str(markers.shape))

markersOnly = markers[:,:,0:39]
print('markersOnly.shape=' + str(markersOnly.shape))

height = markers[:,:,39] 
weight = markers[:,:,40] 
legBin = markers[:,:,41] 
print('height shape = ' + str(height.shape))

# Remove Weight from markers
inds = [] ;

inds2 = range(0,16,2)
inds3 = np.concatenate((range(0,39),[41]))
markers = markers[:,:,inds3]

# Normalize positions by height - markers and COP
markers[:,:,range(39)] = np.divide(markers[:,:,range(39)],np.reshape(height,(height.shape[0],height.shape[1],1)))

# Add velocities and accelerations
vel = np.diff(markers[:,:,range(39)],n=1, axis=1)
vel = np.concatenate((vel,np.zeros((vel.shape[0],1,vel.shape[2]))),axis=1)
acc = np.diff(markers[:,:,range(39)],n=2, axis=1)
acc = np.concatenate((acc,np.zeros((acc.shape[0],2,acc.shape[2]))),axis=1)
markersIn = np.concatenate((markers,vel,acc),axis=-1)
markersIn = markersIn[:,inds2,:]


print('New Markers Shape ' + str(markersIn.shape))


# In[6]:


# Adjust output to correct format
KAM = KAMIn.reshape((-1,30,1))
# Normalize KAM data by height and weight
KAM = np.divide(KAM,weight.reshape(-1,30,1))
KAM = np.divide(KAM,height.reshape(-1,30,1))*100
print(KAM.shape)

# Normalize marker data by height
# markers = np.divide(markersIn,height.reshape(-1,markersIn.shape[1],1))

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

# Build training dataset and labels 
# Doing Peaks Now
inputTrain = markersIn[trainInds,:,:]
labelTrain = np.reshape(KAMP1[trainInds,:,:],(len(trainInds),1))
# inputTrain = inputTrain.reshape((inputTrain.shape[0],-1))

inputDev = markersIn[devInds,:,:]
labelDev = np.reshape(KAMP1[devInds,:,:],(len(devInds),1))
# inputDev = inputDev.reshape((inputDev.shape[0],-1))

inputTest = markersIn[testInds,:,:]
labelTest = np.reshape(KAMP1[testInds,:,:],(len(testInds),1))
# inputTest = inputTest.reshape((inputTest.shape[0],-1))

print('inputTrain size = ' + str(inputTrain.shape))


# Construct Model

# In[8]:


# Initialize our ANN by creating an instance of Sequential. The Sequential function initializes a linear stack of layers.
classifier = Sequential()

def construct_model(hidden = 32, lstm_layers = 2, nTimesteps = 8,nFeatures = 118, output_dim = 2):
    model = Sequential()
#     model.add(Bidirectional(LSTM(input_shape = (input_dim,),input_dim=input_dim, output_dim=hidden, return_sequences=True))
    model.add(Bidirectional(LSTM(hidden, return_sequences=True),input_shape = (nTimesteps,nFeatures)))
    model.add(Bidirectional(LSTM(output_dim = hidden, return_sequences=False)))
    model.add(Dense(output_dim, activation='linear'))
    model.compile(loss=mean_squared_error, optimizer='adam', metrics=['accuracy'])
    return model

model = construct_model(hidden = 32, lstm_layers = 2, nTimesteps = inputTrain.shape[1],nFeatures = inputTrain.shape[2], output_dim = 1)
model.summary()


# Run Model

# In[12]:


history = model.fit(inputTrain, labelTrain, epochs=15, batch_size=32, verbose=2 , validation_data = (inputDev,labelDev))

model.save('LSTM_peakprediction.hdf5')


# Evaluate Performance

# In[9]:


with open('LSTM_peakPrediction.pickle', 'rb') as f:
    model, history = pickle.load(f)


# In[16]:


lossPlt = plt.plot(history.history['loss'])
DevlossPlt = plt.plot(history.history['val_loss'])

plt.ylabel('Mean Squared Error')
plt.xlabel('Epoch Number');
plt.legend(('Training','Dev'))


# In[14]:


# Predict on some dev Data
visualizeRange = range(100)
data =devData;
labels = devLabels;
print(data.shape)
visualizeRange = range(data.shape[0])

y_pred1 = model.predict(data[visualizeRange,:])

truePlot = plt.plot(labels[visualizeRange])
predPlot = plt.plot(y_pred1)
testData.shape

plt.ylabel('KAM Peak Comparison')
plt.xlabel('Step')
plt.legend(('True','Predicted'),loc=4);

plt.savefig('KAM comparisons.png')

plt.figure(2)
plt.plot(labels[visualizeRange,:],y_pred1,'.')
plt.ylabel('Predicted KAM')
plt.xlabel('True KAM')

test_r2 = r2_numpy(testData,testLabels,model)
print(test_r2)


# In[14]:


# model = keras.models.load_model(''FC_peakprediction.hdf5'')


# In[12]:


r2_train, mse_train = r2_numpy(inputTrain,labelTrain,model)
r2_dev, mse_dev = r2_numpy(inputDev,labelDev,model)
r2_test, mse_test = r2_numpy(inputTest,labelTest,model)

print('Train r^2, then rmse = ' + str([r2_train, np.sqrt(mse_train)]))
print('Dev r^2, then rmse = ' + str([r2_dev, np.sqrt(mse_dev)]))
print('Test r^2, then rmse = ' + str([r2_test, np.sqrt(mse_test)]))

