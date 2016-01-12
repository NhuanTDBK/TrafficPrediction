# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import pickle
import sklearn
from sklearn.metrics import mean_squared_error
import theano
import lasagne as ls
from theano import tensor as T
from lasagne.layers import InputLayer, DenseLayer
from lasagne.updates import nesterov_momentum,sgd
from lasagne.nonlinearities import rectify
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import TrainSplit
from pandas import HDFStore
from DemoPyEvolve import PyEvolve
storeResult=HDFStore("storeResultGN.h5")
class NNGridSearch:
    def __init__(self,X_training,y_training,n_sample2,n_test2):
        if(X_training.shape[0]!=y_training.shape[0]):
            print "X_training shape must match y_training shape"
        self.X_training = X_training
        self.y_training = y_training
        self.n_sample2 = n_sample2
        self.n_test2 = n_test2
        print "Generate X_test and y_test"
        print "X_test..."
        print "Multi Layer Perceptron..."
        #Build layer for MLP
        
        print "Neural network initialize"
    def gridsearch_alpha(self,learning_rate,index,params=None):
        hidden_unit = ((index+1)*2)/3
        self.l_in = ls.layers.InputLayer(shape=(None,n_input),input_var=None,W=params.T)
        self.l_hidden = ls.layers.DenseLayer(self.l_in,num_units=15,nonlinearity=ls.nonlinearities.rectify)
        self.network = l_out = ls.layers.DenseLayer(self.l_hidden,num_units=1)
        list_results = np.array([learning_rate.shape[0]],dtype=np.float64)
        for item in learning_rate:
            #Init Neural net
            net1 = NeuralNet(
                layers=self.network,
                # optimization method:
                update=nesterov_momentum,
                update_learning_rate=item,
                update_momentum=0.9,
                regression=True,  # flag to indicate we're dealing with regression problem
                max_epochs=800,  # we want to train this many epochs
#		verbose=1,
                eval_size = 0.4
            )
            #
            
            net1.fit(self.X_training,self.y_training)
            self.pred = net1.predict(self.n_sample2)
            name_file = "GeneticParams/saveNeuralNetwork_%s_%s.tdn" %(item,index)
            net1.save_params_to(name_file)
            score_nn = net1.score(self.n_sample2,self.n_test2)
            list_results[item] = score_nn
            print "index=%f,item=%f,score=%f"%(index,item,score_nn)
        return list_results

# In[ ]:
list_ninput = np.arange(2,21)
learning_rate = np.array([0.00001,0.000001,0.1,0.01,0.001,0.0001])
list_results = np.zeros([ list_ninput.shape[0], learning_rate.shape[0] ],dtype=np.float64)
for i in list_ninput:
    print '.'
    n_input = i
    from __init__ import *
    geneticEngine = PyEvolve(n_input)
    geneticEngine.fit()
    nnParams = geneticEngine.getParam()
    X_training, y_training,n_sample2,n_test2 = get_training(i)
    result = np.zeros(len(learning_rate),dtype=np.float64)
    test = NNGridSearch(X_training,y_training,n_sample2,n_test2)
    list_results[i-2] = test.gridsearch_alpha(learning_rate,i,nnParams)
    
# In[44]:
print "Saving data..."
storeResult["results_gn"] = pd.DataFrame(list_results,index=list_ninput,columns=learning_rate)
storeResult.close()
# ax = pl.subplot()
# ax.set_color_cycle(['blue','red'])
# ax.plot(n_test2,label="actual")
# ax.plot(test.pred,label="predict")
# ax.legend()
# pl.show()
