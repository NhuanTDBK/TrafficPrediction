import numpy as np
import pandas as pd
import sqlite3
import sys
import time
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
        bulk_size = 50000
        start_time = time.time()
        raw_data_chunk = pd.read_csv(raw_data_name,names=["Timestamp"],chunksize=bulk_size)
        for raw_data in raw_data_chunk:
            self.record_traffic_by_data(raw_data)
        end_time = time.time()
        print end_time - start_time
    def record_traffic_by_data(self,raw_data):
        
        # In[ ]:
        # dataCount = np.array(np.zeros(length))
        print "Flush out all data..."
        self.c.execute("DELETE FROM %s"%self.table_name)
        length = raw_data.shape[0]
        tmp_data = raw_data
        try:
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
        dt = self.c.execute('select time,count(time) from %s group by time'%self.table_name).fetchall()
        for item in dt:
            self.c.execute('INSERT INTO %s VALUES (%d,%d)'%(self.table_workload,int(item[0]),int(item[1])))
        self.conn.commit()
    def readFolder(self,folder_name):
        files = listdir(folder_name)
        total = len(files)
        for (index, filename) in enumerate(files):
            print "Reading %s"%filename
            print "Processing %d/%d"%(index+1,total)
            self.record_traffic("%s/%s"%(folder_name,filename))
        self.finalize()
    def finalize(self):
        self.conn.commit()
        self.conn.close()
worker = TrafficExtract()
worker.readFolder(sys.argv[1])


# In[16]:




# In[ ]:



