import numpy as np
import pandas as pd
from pandas import HDFStore
store = HDFStore("storeTraffic.h5")
workload_actual = pd.Series.from_csv("10min_workload.csv",header=None,index_col=None)
def get_training(n_input,n_periodic=0):
 #   n_row = 578 
 # group du lieu
    data = store["connTrain"]
    dataTest = store["connTest"]
    raw_data = store["raw_data_conn"]
    n_row = data.shape[0]
    print "Generate X_traing, y_traing"
    print "X_training loading..."
#     X_training = np.asarray([[data.iloc[t-i-1] for i in range(0,n_input)]
#                  for t in np.arange(n_input,n_row)])
    X_training = []
    max_val = float(store["raw_conn_train"].max())
    min_val = float(store["raw_conn_train"].min())
    for t in range(n_input,n_row):
        temp = []
        for i in range(0,n_input):
            temp.append(data.iloc[t-i-1])
        for j in range(1,n_periodic+1):
            start_idx = data.index[t]
            norVal = (workload_actual[start_idx-142*j]-min_val)/(max_val-min_val)
            temp.append(norVal)
        X_training.append(temp)
    print "y_training loading..."
    y_training = np.asarray(data.iloc[n_input:n_row])
    print "X_test..."
    n_sample2 = X_training
    print "y_test..."
    n_test2 =  np.asarray(dataTest.iloc[n_input:dataTest.shape[0]])
    return np.asarray(X_training), y_training,np.asarray(n_sample2),n_test2

# def get_training_periodic(n_input,n_periodic):
#     print "Hello"
# #   n_row = 578 
#  # group du lieu
# #     data = store["connTrain"]
# #     dataTest = store["connTest"]
# #     raw_data = store["raw_data_conn"]
# #     n_row = data.shape[0]
# #     print "Generate X_traing, y_traing"
# #     print "X_training loading..."
# #     X_training = np.asarray([[data.iloc[t-i-1] for i in range(0,n_input)]
# #                  for t in np.arange(n_input,n_row)])
# #     X_training = []
# #     for t in range(n_input,n_row):
# #         print t
# # #         temp = []
# # #         for i in range(0,n_input):
# # #             temp.append(data.iloc[t-i-1])
# # #         for j in range(1,n_periodic+1):
# # #             temp.append(raw_data[raw_data.index[t]-142*j])
# #     print "y_training loading..."
# #     y_training = np.asarray(data.iloc[n_input:n_row])
# #     print "X_test..."
# #     n_sample2 = np.asarray([[dataTest.iloc[t-i-1] for i in range(0,n_input)]
# #                  for t in np.arange(n_input,dataTest.shape[0])])
# #     print "y_test..."
# #     n_test2 =  np.asarray(dataTest.iloc[n_input:dataTest.shape[0]])
# #     return X_training, y_training,n_sample2,n_test2
