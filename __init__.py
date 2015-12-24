import numpy as np
import pandas as pd
from pandas import HDFStore
store = HDFStore("storeTraffic.h5")
def get_training(n_input):
 #   n_row = 578 
 # group du lieu
    data = store["raw_conn_train"]
    dataTest = store["raw_conn_test"]
    
    n_row = data.shape[0]
    print "Generate X_traing, y_traing"
    print "X_training loading..."
    X_training = np.asarray([[data.iloc[t-i-1] for i in range(0,n_input)]
                 for t in np.arange(n_input,n_row)])
    print "y_training loading..."
    y_training = np.asarray(data.iloc[n_input:n_row])
    print "X_test..."
    n_sample2 = np.asarray([[dataTest.iloc[t-i-1] for i in range(0,n_input)]
                 for t in np.arange(n_input,dataTest.shape[0])])
    print "y_test..."
    n_test2 =  np.asarray(data.iloc[n_input:dataTest.shape[0]])
    return X_training, y_training,n_sample2,n_test2
