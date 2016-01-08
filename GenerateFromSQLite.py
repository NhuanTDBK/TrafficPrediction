
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
from pandas import HDFStore
import sqlite3
import datetime
from datetime import datetime


# In[2]:

# conn = sqlite3.connect('trafficdb')
# raw_data = pd.read_sql("SELECT * FROM workload",conn)
raw_data_conn = pd.Series.from_csv("10min_workload.csv",header=None,index_col=None)


# In[25]:

# # data = pd.read_sql("SELECT count FROM workload where time < 895096802",conn)["count"]
# range_list = len(raw_data_conn)
# jump_list = 600
# init_list = 0;
# length = range_list/jump_list
# # In[ ]:

# dataCount = pd.Series(np.zeros(length))
# print "Count list"
# for i in np.arange(0,length):
#     tmp = raw_data_conn[init_list:init_list+jump_list]
#     dataCount[i] = tmp.sum()
#     init_list+=jump_list


# In[21]:

# ax = pl.subplot()
# ax.set_color_cycle(['blue','red','green'])
# # ax.plot(gn_pred,label="Genetic Neural Network")
# t = np.arange(0,raw_data_conn.shape[0],142)
# ax.plot(raw_data_conn,label="Actual")
# for i in np.arange(0,raw_data_conn.shape[0],142):
#     ax.axvline(i,color='r')
# ax.legend()
# pl.show()


# In[23]:

# ex=raw_data_conn[142*40:142*46]
# ax = pl.subplot()
# ax.set_color_cycle(['blue','red','green'])
# # ax.plot(gn_pred,label="Genetic Neural Network")
# t = np.arange(0,ex.shape[0],142)
# ax.plot(ex,label="Actual")
# for i in np.arange(0,ex.shape[0],142):
#     ax.axvline(i,color='r')
# ax.legend()
# pl.show()


# In[12]:

from pandas import HDFStore
def normalize(dataCount):
    dataNorm = pd.Series(np.zeros(dataCount.shape[0]),dtype=np.float64)
    dataNorm = (dataCount - dataCount.min())/(dataCount.max()-dataCount.min())
    return dataNorm
store = HDFStore("storeTraffic.h5")
# # In[62]:
#dataTraining = raw_data_conn[142*3:142*7]
dataTraining = raw_data_conn[142*40:142*46]
# print "Loading storage"
store["connTrain"]=normalize(pd.Series(dataTraining))
store["connTest"] = normalize(pd.Series(dataTraining))
store["raw_conn_train"]=pd.Series(dataTraining)
store["raw_conn_test"] = pd.Series(dataTraining)

store.close()

