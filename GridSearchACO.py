from SplitTrainTest import SplitTrainTest
from sklearn.grid_search import GridSearchCV
from initializer import *
from TrafficFeeder import TrafficFeeder
from ACOEstimator import ACOEstimator
param_dicts = {
    "Q":np.arange(0.01,0.1,step=0.01),
    "epsilon":np.arange(0.1,1.0,step=0.05)
}
n_sliding_window = 4
n_periodic = 1
n_input = n_sliding_window + n_periodic
neural_shape=[n_input,15,1]
dataFeeder = TrafficFeeder()
X,y = dataFeeder.fetch_traffic_training(n_sliding_window,n_periodic,(40,46))
estimator = ACOEstimator()
archive_solution = construct_solution(estimator.number_of_solutions,neural_shape)
cv = SplitTrainTest(X.shape[0])
fit_param = {'neural_shape':neural_shape,"archive":archive_solution}
# estimator.fit(X,y,**fit_param)
gs = GridSearchCV(estimator,param_grid=param_dicts,cv=cv,n_jobs=-1,fit_params=fit_param,scoring='mean_squared_error')
gs.fit(X,y)
print gs.best_estimator_
print gs.best_estimator_.score(X,y)
