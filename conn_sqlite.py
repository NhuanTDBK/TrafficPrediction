
# coding: utf-8

# In[36]:

import numpy as np
import pandas as pd
import sqlite3
from os import listdir


# In[31]:

conn = sqlite3.connect('trafficdb')
c = conn.cursor()
table_name = "timetraffic"
table_workload = "workload"
c.execute("DELETE FROM %s"%table_workload)


# In[32]:

# raw_data_name = sys.argv[1]
def record_traffic(raw_data_name):
    raw_data = pd.read_csv(raw_data_name)
    length = raw_data.shape[0]
    # In[ ]:
    # dataCount = np.array(np.zeros(length))
    print "Flush out all data..."
    c.execute("DELETE FROM %s"%table_name)
    for i in np.arange(0,length):
        index = raw_data.irow(i)["Timestamp"]
        c.execute('INSERT INTO %s VALUES (%d,%d)'%(table_name,index,1))
    dt = c.execute('select timestamp,count(timestamp) from %s group by timestamp'%table_name).fetchall()
    for item in dt:
        c.execute('INSERT INTO %s VALUES (%d,%d)'%(table_workload,item[0],item[1]))


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

files = listdir(sys.argv[1])
for filename in files:
    print "Reading %s"%filename
    record_traffic(filename)


# In[6]:

conn.commit()


# In[16]:

conn.close()


# In[ ]:



