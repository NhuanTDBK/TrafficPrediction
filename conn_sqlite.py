
# coding: utf-8

# In[5]:

import numpy as np
import pandas as pd
import sqlite3
import sys
from os import listdir


# In[16]:

class TrafficExtract():
    def __init__(self):
        self.conn = sqlite3.connect('trafficdb')
        self.c = self.conn.cursor()
        self.table_name = "timetraffic"
        self.table_workload = "workload"
        self.c.execute("DELETE FROM %s"%self.table_workload)
    def record_traffic(self,raw_data_name):
        print "Processing %s"%raw_data_name
        raw_data = pd.read_csv(raw_data_name,low_memory=False)
        length = raw_data.shape[0]
        # In[ ]:
        # dataCount = np.array(np.zeros(length))
        print "Flush out all data..."
        self.c.execute("DELETE FROM %s"%self.table_name)
        for i in np.arange(0,length):
	    try:
 	           index = raw_data.irow(i)["Timestamp"]
	           self.c.execute('INSERT INTO %s VALUES (%d,%d)'%(self.table_name,int(index),1))
	    except Exception as e:
		   print index
		   pass			
        dt = self.c.execute('select timestamp,count(timestamp) from %s group by timestamp'%self.table_name).fetchall()
        for item in dt:
            self.c.execute('INSERT INTO %s VALUES (%d,%d)'%(self.table_workload,int(item[0]),int(item[1])))
    def readFolder(self,folder_name):
        files = listdir(folder_name)
        for filename in files:
            print "Reading %s"%filename
            self.record_traffic("%s/%s"%(folder_name,filename))
        self.conn.commit()
	self.finalize()
    def finalize(self):
        self.conn.close()


# In[12]:

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


# In[13]:


#     c.execute('INSERT INTO %s VALUES (%d,%d)'%(table_workload,i,tmp_sum))
# start_idx = 0;
# jump_list = 600;
# size = len(list)/jump_list
# dataCount = np.zeros(size)
# for i in np.arange(size):
#     tmp_sum = sum([pair[0] for pair in list[start_idx:start_idx+jump_list]])
#     start_idx += jump_list
#     c.execute('INSERT INTO %s VALUES (%d,%d)'%(table_workload,i,tmp_sum))


# In[14]:

# files = listdir(sys.argv[1])
# for filename in files:
#     print "Reading %s"%filename
#     record_traffic(filename)


# In[17]:

worker = TrafficExtract()
worker.readFolder(sys.argv[1])


# In[16]:




# In[ ]:



