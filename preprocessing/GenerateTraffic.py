
# coding: utf-8

# In[46]:

import pandas as pd
import numpy as np
import sys
from pandas import HDFStore

def normalize(dataCount):
    dataNorm = pd.Series(np.zeros(dataCount.shape[0]),dtype=np.float64)
    dataNorm = (dataCount - dataCount.min())/(dataCount.max()-dataCount.min())
    return dataNorm

# In[53]:

raw_data_name = sys.argv[1]
#"ita_public_tools/output/data.csv"
raw_data = pd.read_csv("/home/nhuan/Tools/ita_public_tools/output/data.csv",names=["Timestamp"])
store = HDFStore("storeTraffic.h5")
# In[62]:
print "Loading storage"
data = raw_data.groupby('Timestamp').count()["Timestamp"]
print "generate data"
# Config variable
range_list = data.shape[0]
jump_list = 600
init_list = 0;
length = data.shape[0]/jump_list
# In[ ]:

dataCount = np.array(np.zeros(length))
print "Count list"
for i in np.arange(0,length):
    tmp = data[init_list:init_list+jump_list]
    dataCount[i] = tmp.sum()
    init_list+=jump_list
print "Saving..."
store["connTrain"]=normalize(pd.Series(dataCount))
# raw_data_name = "ita_public_tools/output/data.csv"
# raw_data = pd.read_csv(raw_data_name)
# store = HDFStore("storeTraffic.h5")
# data = raw_data.groupby('Timestamp').count()["Timestamp"]
store["connTest"] = normalize(pd.Series(dataCount))
store["raw_conn_train"]=pd.Series(dataCount)
store["raw_conn_test"] = pd.Series(dataCount)


