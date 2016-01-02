
# coding: utf-8

# In[1]:

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
storeResult=HDFStore("storeResult.h5")


# In[ ]:

n_input = 16
n_gen = 400
score = 1
score_min = 1
gen_min = 400
genetic = PyEvolve(n_input)
while((score>=0.001) or (n_gen<100000)):
    print n_gen
    score = genetic.fit(n_gen)
    print score
    if(score_min>score):
        score_min = score
    if(gen_min>n_gen):
        gen_min = n_gen
    n_gen += 100
print score_min
print gen_min


# In[ ]:



