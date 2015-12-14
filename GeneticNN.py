
# coding: utf-8

# In[1]:



import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import pickle as pkl
import sklearn
from sklearn.metrics import mean_squared_error
import theano
import lasagne as ls
from theano import tensor as T
from lasagne.layers import InputLayer, DenseLayer
from lasagne.updates import nesterov_momentum
from lasagne.nonlinearities import rectify
from nolearn.lasagne import NeuralNet
from __init__ import *


# In[ ]:


#Init data training
print "Reading file......"
X_training, y_training,n_sample2,n_test2 = get_training()
if(X_training.shape[0]!=y_training.shape[0]):
    print "X_training shape must match y_training shape"
print "Generate X_test and y_test"
n_input = 11
print "X_test..."
#n_sample2 = np.asarray([[raw_data.ix[t-i][4] for i in range(1,n_input)] for t in np.arange (289*400,289*410)])
print "y_test..."
#n_test2 =  np.asarray([raw_data.ix[t][4] for t in np.arange(289*400+1,289*410+1)])
print "Multi Layer Perceptron..."
parameter = pkl.load(open('saveGenetic.p', 'rb'))
#Build layer for MLP
l_in = ls.layers.InputLayer(shape=(None,10),input_var=None,W=parameter.T)
l_hidden = ls.layers.DenseLayer(l_in,num_units=10,nonlinearity=ls.nonlinearities.rectify)
network = l_out = ls.layers.DenseLayer(l_hidden,num_units=1)
print "Neural network initialize"
#Init Neural net
net1 = NeuralNet(
    layers=network,
    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,
    regression=True,  # flag to indicate we're dealing with regression problem
    max_epochs=400,  # we want to train this many epochs
    verbose=1,
)
#
print "Training time!!!!!....."
net1.fit(X_training,y_training)
net1.save_params_to("saveNeuralNetworkv2.tdn")
print "Score rate = "
print net1.score(n_sample2,n_test2)
print net1.predict(n_sample2)[0:2]

