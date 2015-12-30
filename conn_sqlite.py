
# coding: utf-8

# In[5]:

import numpy as np
import pandas as pd
import sqlite3
import sys
import timeit
from os import listdir

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
        bulk_insert_size = 50000
        # In[ ]:
        # dataCount = np.array(np.zeros(length))
        print "Flush out all data..."
        self.c.execute("DELETE FROM %s"%self.table_name)
        try:
            tmp_data = raw_data
            mapFunction = lambda x: (int(x),1)
            list_count =[(item[0][0],item[0][1]) for item in tmp_data.applymap(mapFunction).values.tolist()]
            self.conn.executemany("INSERT INTO %s(timestamp,count) VALUES (?,?)"%self.table_name,list_count)
        except Exception as ex:
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
        self.finalize()
    def readFolder(self,folder_name):
        files = listdir(folder_name)
        total = len(files)
        for (index, filename) in enumerate(files):
            print "Reading %s"%filename
            print "Processing %d/%d"%(index,total)
            self.record_traffic("%s/%s"%(folder_name,filename))
    def finalize(self):
        self.conn.commit()
        self.conn.close()

worker = TrafficExtract()
worker.readFolder(sys.argv[1])


# In[16]:




# In[ ]:



