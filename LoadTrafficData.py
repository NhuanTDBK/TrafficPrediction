
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import pandas as pd
from pandas import HDFStore


# In[2]:

store = HDFStore("storeTraffic.h5")
store


# In[4]:

connTrain = np.array(store["raw_conn_train"])


# In[8]:



