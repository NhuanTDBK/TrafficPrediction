from TrafficFeeder import *
from AntOptimizationContinous import AntOptimization
from FeedFlow import FeedFlow
from NeuralFlow import NeuralFlowRegressor
from sklearn.metrics import mean_squared_error
temp = []
n_hidden_layer = 15
n_output = 1
summary_writter = []
data_feeder = TrafficFeeder()
def model_select(index):
    i = 46
    skip_list = 3
    n_periodic = 1
    n_input = index
    neural_shape = [n_input+n_periodic,n_hidden_layer,n_output]
    X_test,y_test = data_feeder.fetch_traffic_training(n_input,n_periodic,(i, i + skip_list))
    X_training,y_training = data_feeder.generate((40, 46))
    feedflow_model = FeedFlow(neural_shape,uniform_init=True)
    ACOEngigne = AntOptimization(feedflow_model.neural_shape,feedflow_model.total_nodes,number_of_solutions=200,max_epochs=30)
    weights_matrix = ACOEngigne.optimize(feedflow_model,summary_writter,X_training,y_training)
    neuralFlow = NeuralFlowRegressor(neural_shape,weights_matrix=weights_matrix)
    neuralFlow.fit(X_training,y_training)
    y_pred_index = data_feeder.convert(neuralFlow.predict(X_test))
    score_index = np.sqrt(mean_squared_error(y_pred_index, y_test))
    print score_index
    temp.append([index, score_index])
    neuralFlow.save('param_ants%s'%index)
# index = 4
# for index in np.arange(3,20):
#     model_select(index)
# print temp
# test_data = pd.DataFrame(temp,index=np.arange(3,20),columns=["index","score"])
# score = test_data["score"].min()
# idx = test_data["score"].argmin()
# print score,idx
from sklearn.grid_search import GridSearchCV

# index =n_input= idx
# n_periodic = 1
# i = 46
# skip_list = 3
# neuralFlow = NeuralFlow(index,1)
# neuralFlow.net1.load_params_from("param_ant/param%s"%index)
# sample_data = LoadParam(None,0,n_input,n_periodic)
# X_test,y_test = sample_data.generate_test((i,i+skip_list))
# y_pred = sample_data.convert(neuralFlow.net1.predict(X_test))
# # print np.sqrt(mean_squared_error(y_test,y_pred))
# ax = pl.subplot()
# ax.plot(y_pred)
# ax.plot(y_test)
# # ax.plot(summary_writter)
# pl.show()



















