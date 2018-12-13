
# coding: utf-8

# In[1]:


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
from keras.layers import Dense, LSTM, TimeDistributed, BatchNormalization, Input, Dropout
import keras.backend as K
from keras.losses import mean_squared_error
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import mpld3
import pickle
#mpld3.enable_notebook()

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
    mse = np.mean(np.square(y_pred2-labels))
    return r2, mse


# Prepare the data

# In[2]:


config = tf.ConfigProto(device_count = {'CPU' : 1, 'GPU' : 0})
session = tf.Session(config=config)
K.set_session(session)


# In[3]:


# Import data
markersIn = sio.loadmat('markersWithCOP2.mat')
KAM = sio.loadmat('KAMWithCOP2.mat')
subind = sio.loadmat('subindWithCOP2.mat')

# Define input (markers) and output (KAM)
markersInput = markersIn['markers']
KAMIn = KAM['KAM']
subind = subind['subind']
subind = subind -1 # Change to python indexing


# In[4]:


markers = np.moveaxis(markersInput, [0,1,2], [2,1,0])
print('Original markers.shape=' + str(markers.shape))

markersOnly = markers[:,:,range(39)]
print('markersOnly.shape=' + str(markersOnly.shape))

weight = markers[:,:,121] 
height = markers[:,:,122] 
legBin = markers[:,:,123] 
print('height shape = ' + str(height.shape))

# Remove Weight from markers
inds = [] ;

inds2 = range(0,16,2)
# inds3 = np.concatenate((range(0,121),[117 118 119 120 123])) # add leg to markers
inds3 = np.concatenate((range(0,39),[123])) # add leg to markers
markers = markers[:,:,inds3]

# normalize positions by height - markers and COP
markers[:,:,range(39)] = np.divide(markers[:,:,range(39)],np.reshape(height,(height.shape[0],height.shape[1],1)))

# # Add velocities and accelerations
vel = np.diff(markers[:,:,range(39)],n=1, axis=1)
vel = np.concatenate((vel,np.zeros((vel.shape[0],1,vel.shape[2]))),axis=1)
acc = np.diff(markers[:,:,range(39)],n=2, axis=1)
acc = np.concatenate((acc,np.zeros((acc.shape[0],2,acc.shape[2]))),axis=1)
markers = np.concatenate((markers,vel,acc),axis=-1)
markersIn = markers[:,inds2,:]

print('New Markers Shape ' + str(markersIn.shape))


# In[5]:


# Adjust output to correct format
KAM = KAMIn.reshape((-1,30,1))
# Normalize KAM data by height and weight
KAM = np.divide(KAM,weight.reshape(-1,30,1))
KAM = np.divide(KAM,height.reshape(-1,30,1))*100
print(KAM.shape)

# Normalize marker data by height
# markers = np.divide(markersIn,height[:,inds2].reshape(-1,markersIn.shape[1],1))
markers = markersIn

KAMP1 = np.max(KAM[:,range(15),:],axis=1)
KAMP1 = KAMP1.reshape((-1,1,1))
print(KAMP1.shape)

print(markers.shape)


# In[6]:


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

# Build training dataset and labels # Doing Peaks Now
inputTrain = markers[trainInds,:,:]
labelTrain = KAMP1[trainInds,:,:]
inputTrain = inputTrain.reshape((inputTrain.shape[0],-1))

inputDev = markers[devInds,:,:]
labelDev = KAMP1[devInds,:,:]
inputDev = inputDev.reshape((inputDev.shape[0],-1))

inputTest = markers[testInds,:,:]
labelTest = KAMP1[testInds,:,:]
inputTest = inputTest.reshape((inputTest.shape[0],-1))

print('inputTrain size = ' + str(inputTrain.shape))


# In[8]:


# Flatten Input and Output Data
mTrain = inputTrain.shape[0] # number of examples to train on 
# mTrain = 1000

#Input is mx328
trainData = inputTrain[0:mTrain,:]
devData = inputDev
testData = inputTest

print("trainData shape = " + str(trainData.shape))


#Output is mx1
LabelsFull = np.reshape(labelTrain,(labelTrain.shape[0]*labelTrain.shape[1],labelTrain.shape[2]))
trainLabels = LabelsFull[0:mTrain]
devLabelsFull = np.reshape(labelDev,(labelDev.shape[0]*labelDev.shape[1],labelDev.shape[2]))
devLabels = devLabelsFull
testLabelsFull = np.reshape(labelTest,(labelTest.shape[0]*labelTest.shape[1],labelTest.shape[2]))
testLabels = testLabelsFull

print("trainLabels shape = " + str(trainLabels.shape))


# Construct Model

# In[9]:


# Build the model
train_r2 = np.empty((1000,1))
dev_r2 = np.empty((1000,1))
train_loss = np.empty((1000,1))
dev_loss = np.empty((1000,1))
epochCount = 0 ;

def construct_model(nHiddenUnits = 32, nHiddenLayers = 15, input_dim = 41, output_dim = 1):
    model = Sequential()
    model.add(Dense(800,input_shape = (input_dim,), kernel_initializer=glorot_normal(seed=None) , activation='relu')) #,kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01)))
    for i in range(nHiddenLayers-1):
        model.add(Dropout(0.01))
        model.add(Dense(nHiddenUnits , kernel_initializer=glorot_normal(seed=None) , activation='relu')) #,kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01)))
    
    model.add(Dropout(0.01))
    model.add(Dense(1,kernel_initializer=glorot_normal(seed=None),activation='linear'))
    model.compile(loss='mean_squared_error',optimizer='adam')
    return model

model = construct_model(nHiddenUnits = 100, nHiddenLayers = 1, input_dim = trainData.shape[1], output_dim = trainLabels.shape[1])
model.summary()


# Run Model

# In[39]:


nEpochs = 4

for i in range(nEpochs):
    print('Epoch ' + str(i+1) + ' of ' + str(nEpochs) + '.')

    history = model.fit(trainData,trainLabels, epochs=1 , batch_size = 32, shuffle = True, verbose=2)
    
    train_r2[epochCount], train_loss[epochCount] = r2_numpy(trainData,trainLabels,model)
    test_r2, test_loss = r2_numpy(testData,testLabels,model)
    dev_r2[epochCount], dev_loss[epochCount] = r2_numpy(devData,devLabels,model)
    print('Train_loss = ' + str(train_loss[epochCount]) + ', Train_r2 = ' + str(train_r2[epochCount]) + ', Dev_loss = ' + str(dev_loss[epochCount]) + ', Dev_r2 = ' + str(dev_r2[epochCount]) + ', Test_r2 = ' + str(test_r2))
    
    epochCount = epochCount + 1 ;
model.save('FC_peakprediction.hdf5')


# Evaluate Performance

# In[40]:


lossPlt = plt.plot(train_loss[range(epochCount)])
DevlossPlt = plt.plot(dev_loss[range(epochCount)])

plt.ylabel('Mean Squared Error')
plt.xlabel('Epoch Number');
plt.legend(('Training','Dev'))

plt.figure(2)
r2Plt = plt.plot(train_r2[range(epochCount)])
devr2Plt = plt.plot(dev_r2[range(epochCount)])
plt.ylim([.5, 1])
plt.ylabel('r^2')
plt.xlabel('Epoch Number');
plt.legend(('Training','Dev'))


# Save Things

# Load Things

# In[10]:


with open('theBestModel.pickle', 'rb') as f:
    model, history,metrics = pickle.load(f)

    train_r2,train_loss,dev_r2,dev_loss = metrics


# In[24]:


# Predict on some dev Data
visualizeRange = range(100)
data =testData;
labels = testLabels;
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
plt.axis('equal')
plt.plot([1,6],[1,6],'k')

test_r2 = r2_numpy(testData,testLabels,model)
print(test_r2)


# In[61]:


model2 = keras.models.load_model('FC_peakprediction.hdf5', custom_objects={'r2_keras': r2_keras})


# In[60]:


subindTrain = subind[0,trainInds]
subindDev = subind[0,devInds]
subindTest = subind[0,testInds]



# Define Colormap
cmap = plt.cm.get_cmap('jet')

rgba = cmap(0.5)


sInd = subindDev[0]
i = 0 ;

for j in range(len(devNums)):
    colorVal = cmap(j/len(devNums))
    sInd = subindDev[i]
    print(j)
    while  sInd == subindDev[i]:
        if i<len(subindDev)-1:
            plt.plot(labels[visualizeRange[i],:],y_pred1[i],'.',color=colorVal)
            i = i+1

plt.axis('equal')


plt.ylabel('Predicted KAM')
plt.xlabel('True KAM')


# In[11]:


r2_train, mse_train = r2_numpy(trainData,trainLabels,model)
r2_dev, mse_dev = r2_numpy(devData,devLabels,model)
r2_test, mse_test = r2_numpy(testData,testLabels,model)

print('Train r^2, then rmse = ' + str([r2_train, np.sqrt(mse_train)]))
print('Dev r^2, then rmse = ' + str([r2_dev, np.sqrt(mse_dev)]))
print('Test r^2, then rmse = ' + str([r2_test, np.sqrt(mse_test)]))

