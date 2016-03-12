
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import lasagne as ls
from nolearn.lasagne import NeuralNet
from pandas import HDFStore
store = HDFStore("storeTraffic.h5")


# In[2]:

n_input = 18
def initNN(n_input):
    #Build layer for MLP
    l_in = ls.layers.InputLayer(shape=(None,n_input),input_var=None)
    l_hidden = ls.layers.DenseLayer(l_in,num_units=15,nonlinearity=ls.nonlinearities.rectify)
    network = l_out = ls.layers.DenseLayer(l_hidden,num_units=1)
    print "Neural network initialize"
    #Init Neural net
    net1 = NeuralNet(
        layers=network,
        # optimization method:
        update=nesterov_momentum,
        update_learning_rate=0.000001,
        update_momentum=0.9,
        regression=True,  # flag to indicate we're dealing with regression problem
        max_epochs=1000,  # we want to train this many epochs
        verbose=1,
    )
    return net1


# In[3]:

net1 = initNN(n_input)
net1.load_params_from('Params/saveNeuralNetwork_0.0001_%s.tdn'%n_input)
net2 = initNN(n_input)
net2.load_params_from('GeneticParams/saveNeuralNetwork_0.0001_%s.tdn'%n_input)


# In[4]:

maxC = store["connTotal"].max()
minC = store["connTotal"].min()
def convert(input):
    return (input*(maxC-minC)+minC)
def normalize(dataCount):
    dataNorm = pd.Series(np.zeros(dataCount.shape[0]),dtype=np.float64)
    dataNorm = (dataCount - dataCount.min())/(dataCount.max()-dataCount.min())
    return dataNorm


# In[56]:

final_day = 1
dataReal = store["connTotal"][final_day*142-n_input:final_day*142*2]
data = normalize(dataReal)
start_idx = data.index[0]
end_idx = data.index[data.shape[0]-1]
# print "X_training loading..."
X_training = np.asarray([[ data[t+i] for i in range(0,n_input)] for t in np.arange(start_idx,end_idx-n_input+1)])
# print "y_training loading..."
y_training = np.asarray(dataReal.iloc[n_input:end_idx-start_idx+1])
# y_training[0:3], gn_pred[0:3]


# In[57]:

nn_pred = convert(net1.predict(X_training))
gn_pred = convert(net1.predict(X_training))


# In[58]:

ax = pl.subplot()
ax.set_color_cycle(['blue','red','green'])
# ax.plot(y_training,label="actual")
ax.plot(nn_pred,label="Simple Neural Network")
ax.plot(gn_pred,label="Genetic Neural Network")
ax.plot(y_training,label="Actual")
ax.legend()
pl.show()


# In[27]:

dataNN = HDFStore("storeResult.h5")["results"]
dataGN = HDFStore("storeResultGenetic.h5")["results"]


# In[28]:

dataNN.icol(-1).order()


# In[29]:

dataGN.icol(-1).order()


# In[43]:

dataTraffic = HDFStore("storeTraffic.h5")["raw_conn_train"]
pattern = np.zeros(len(dataTraffic))
# np.correlate(dataTraffic[0:142],dataTraffic[n:n+142])[0]
n=142
for i in np.arange(len(dataTraffic)):
    pattern[i] = np.correlate(dataTraffic[0:142*3],dataTraffic[n:n+142*3])[0]
    n = n+1
    print pattern[i]


# In[44]:

ax = pl.subplot()
ax.set_color_cycle(['blue','red'])
# ax.plot(y_training,label="actual")
ax.plot(pattern,label="Auto correlation")
ax.legend()
pl.show()


# In[81]:

dataTraffic = HDFStore("storeTraffic.h5")["raw_conn_train"]
pattern = np.zeros(len(dataTraffic))
# np.correlate(dataTraffic[0:142],dataTraffic[n:n+142])[0]
n=142
for i in np.arange(len(dataTraffic)):
    pattern[i] = np.correlate(dataTraffic[0:142*3],dataTraffic[n:n+142*3])[0]
    n = n+1
#     print pattern[i]


# In[82]:

ax = pl.subplot()
ax.set_color_cycle(['blue','red'])
# ax.plot(y_training,label="actual")
ax.plot(pattern,label="Auto correlation")
ax.legend()
pl.show()


# In[48]:

import scipy as sc


# In[60]:

period = sc.signal.periodogram(dataTraffic,fs=6)
period


# In[79]:

f, Pxx_den = sc.signal.periodogram(dataTraffic, fs=6*24*7)
# pl.semilogy(f, Pxx_den)
pl.xlabel('frequency [Hz]')
# pl.ylim([1, 1e2])
pl.ylabel('PSD [V**2/Hz]')
pl.plot(f,Pxx_den)
pl.show()


# In[78]:

ax = pl.subplot()
ax.set_color_cycle(['blue','red'])
# ax.plot(y_training,label="actual")
ax.plot(dataTraffic[0:142*2],label="Auto correlation")
ax.legend()
pl.show()


# In[ ]:

resultNN = HDFStore("")

