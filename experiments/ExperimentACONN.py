import matplotlib.pyplot as plt
import numpy as np
from estimators.ACOEstimator import ACOEstimator
from sklearn.metrics import mean_squared_error

from data.TrafficFeeder import TrafficFeeder
from estimators.NeuralFlow import NeuralFlowRegressor


def plot_fig(y_pred,y_actual,label=["Predict","Actual"]):
    ax= plt.subplot()
    ax.plot(y_pred,label=label[0])
    ax.plot(y_actual,label=label[1])
    ax.legend()
    plt.show()
if __name__ == '__main__':
    best_estimator = None
    best_score = np.Inf
    for loop in np.arange(1,20):
        n_input = 4
        n_periodic = 1
        n_hidden = 15
        neural_shape = [n_input+n_periodic,n_hidden,1]
        Q = 0.09
        epsilon = 0.55

        dataFeeder = TrafficFeeder()
        X_train,y_train = dataFeeder.fetch_traffic_training(n_input,1,(40,46))
        X_test,y_test = dataFeeder.fetch_traffic_test(n_input,1,(46,48))
        # retrieve = [n_input+1,(X_train,y_train,X_test,y_test)]
        acoNet = ACOEstimator(Q=Q,epsilon=epsilon)
        fit_param = {
            "neural_shape":neural_shape
        }
        acoNet.fit(X_train,y_train,**fit_param)
        fit_param["weights_matrix"] = acoNet.best_archive
        neuralNet = NeuralFlowRegressor()
        neuralNet.fit(X_train,y_train,**fit_param)
        y_pred = dataFeeder.convert(neuralNet.predict(X_test))
        score =  np.sqrt(mean_squared_error(y_pred,y_test))
        if(score<best_score):
            best_estimator = acoNet
            print score
        # plot_fig(y_pred,y_test)
    print best_score,best_estimator

