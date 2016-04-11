import matplotlib.pyplot as plt
import numpy as np
from GAEstimator import GAEstimator
from sklearn.metrics import mean_squared_error

from TrafficFeeder import TrafficFeeder
from estimators.NeuralFlow import NeuralFlowRegressor


def plot_fig(y_pred,y_actual,label=["Predict","Actual"]):
    ax= plt.subplot()
    ax.plot(y_pred,label=label[0])
    ax.plot(y_actual,label=label[1])
    ax.legend()
    plt.show()
if __name__ == '__main__':
    n_input = 4
    n_periodic = 1
    n_hidden = 15
    neural_shape = [n_input+n_periodic,n_hidden,1]

    cross_rate = 0.6
    mutation_rate = 0.04
    pop_size = 50

    dataFeeder = TrafficFeeder()
    X_train,y_train = dataFeeder.fetch_traffic_training(n_input,1,(40,46))
    X_test,y_test = dataFeeder.fetch_traffic_test(n_input,1,(46,48))
    # retrieve = [n_input+1,(X_train,y_train,X_test,y_test)]
    gaEstimator = GAEstimator(cross_rate=cross_rate,mutation_rate=mutation_rate,pop_size=pop_size)
    fit_param = {
        "neural_shape":neural_shape
    }
    gaEstimator.fit(X_train,y_train,**fit_param)
    fit_param["weights_matrix"] = gaEstimator.best_archive
    neuralNet = NeuralFlowRegressor()
    neuralNet.fit(X_train,y_train,**fit_param)
    y_pred = dataFeeder.convert(neuralNet.predict(X_test))
    print np.sqrt(mean_squared_error(y_pred,y_test))
    plot_fig(y_pred,y_test)


