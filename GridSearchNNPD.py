import numpy as np
from TrafficFeeder import TrafficFeeder
from NeuralFlow import NeuralFlowRegressor
from joblib import Parallel,delayed
def get_params():
    dataFeeder = TrafficFeeder()
    out = Parallel(n_jobs=-1)(delayed(put_queue)
                        (n_input,dataFeeder) for n_input in range(4,21))
    return out
def put_queue(n_input,dataFeeder):
    X_train,y_train = dataFeeder.fetch_traffic_training(n_input,1,(40,46))
    X_test,y_test = dataFeeder.fetch_traffic_training(n_input,1,(46,48))
    retrieve = [n_input+1,(X_train,y_train,X_test,y_test)]
    return retrieve
def model_fit(param):
    print "Training %s"%param[0]
    neural_shape = [param[0],15,1]
    X_train = param[1][0]
    y_train = param[1][1]
    X_test = param[1][2]
    y_test = param[1][3]
    fit_param = {
        "neural_shape":neural_shape
    }
    neuralNet = NeuralFlowRegressor()
    neuralNet.fit(X_train,y_train,**fit_param)
    return param[0],np.sqrt(neuralNet.score(X_test,y_test))
def fit_and_evaluate():
    out = get_params()
    result = Parallel(n_jobs=-1)(delayed(model_fit)(param) for param in out)
    result_sorted = sorted(result,key=lambda x:x[1])
    # print result_sorted[0]
    return result_sorted
print fit_and_evaluate()