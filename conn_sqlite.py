
# coding: utf-8

# In[36]:

import numpy as np
import pandas as pd
import sqlite3
from os import listdir


# In[31]:

class TrafficExtract():
    def __init__(self):
        self.conn = sqlite3.connect('trafficdb')
        self.c = conn.cursor()
        self.table_name = "timetraffic"
        self.table_workload = "workload"
        self.c.execute("DELETE FROM %s"%table_workload)
    def record_traffic(self,raw_data_name):
        raw_data = pd.read_csv(raw_data_name)
        length = raw_data.shape[0]
        # In[ ]:
        # dataCount = np.array(np.zeros(length))
        print "Flush out all data..."
        self.c.execute("DELETE FROM %s"%table_name)
        for i in np.arange(0,length):
            index = raw_data.irow(i)["Timestamp"]
            self.c.execute('INSERT INTO %s VALUES (%d,%d)'%(table_name,index,1))
        dt = self.c.execute('select timestamp,count(timestamp) from %s group by timestamp'%table_name).fetchall()
        for item in dt:
            self.c.execute('INSERT INTO %s VALUES (%d,%d)'%(table_workload,item[0],item[1]))
    def readFolder(self,folder_name):
        files = listdir(folder_name)
        for filename in files:
            print "Reading %s"%filename
            self.record_traffic("%s/%s"%(folder_name,filename))
        self.conn.commit()
    def finalize(self):
        self.conn.close()


# In[32]:

# raw_data_name = sys.argv[1]
# def record_traffic(raw_data_name):
#     raw_data = pd.read_csv(raw_data_name)
#     length = raw_data.shape[0]
#     # In[ ]:
#     # dataCount = np.array(np.zeros(length))
#     print "Flush out all data..."
#     self.c.execute("DELETE FROM %s"%table_name)
#     for i in np.arange(0,length):
#         index = raw_data.irow(i)["Timestamp"]
#         self.c.execute('INSERT INTO %s VALUES (%d,%d)'%(table_name,index,1))
#     dt = self.c.execute('select timestamp,count(timestamp) from %s group by timestamp'%table_name).fetchall()
#     for item in dt:
#         self.c.execute('INSERT INTO %s VALUES (%d,%d)'%(table_workload,item[0],item[1]))


# In[35]:


#     c.execute('INSERT INTO %s VALUES (%d,%d)'%(table_workload,i,tmp_sum))
# start_idx = 0;
# jump_list = 600;
# size = len(list)/jump_list
# dataCount = np.zeros(size)
# for i in np.arange(size):
#     tmp_sum = sum([pair[0] for pair in list[start_idx:start_idx+jump_list]])
#     start_idx += jump_list
#     c.execute('INSERT INTO %s VALUES (%d,%d)'%(table_workload,i,tmp_sum))


# In[ ]:

# files = listdir(sys.argv[1])
# for filename in files:
#     print "Reading %s"%filename
#     record_traffic(filename)


# In[6]:




# In[16]:




# In[ ]:



