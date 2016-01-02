from pyevolve import GSimpleGA
from pyevolve import G1DList
from pyevolve import Selectors
from pyevolve import Initializators, Mutators
from pyevolve import Crossovers
from lasagne.nonlinearities import rectify,sigmoid
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as pl
from __init__ import *

class PyEvolve:
        
    def __init__(self,n_input):
        self.n_input = n_input
        self.X_data, self.y_data,self.x_test,self.y_test = get_training(n_input)
        self.fan_in = n_input
	self.fan_out = 15
        self.theta_shape = (self.n_input,1)
    def activation(self,x):
    #         return 1.0 / (1 + np.exp(-x))
            result = np.zeros([x.shape[0]],dtype=np.float64)
            for index,e in enumerate(x):
                result[index]=rectify(e)
            return result
    def costFunction(self,X,y,theta):
            m = float(len(X))
            hThetaX = np.array(self.activation(np.dot(X,theta)))
            return np.sum(np.abs(y-hThetaX))
    def eval_score(self,chronosome):
            theta = np.zeros(self.theta_shape)
            for i in np.arange(self.theta_shape[0]) :
                theta[i] = chronosome[i]
            return self.costFunction(self.X_data,self.y_data,theta);
    def fit(self,gen=400,freq_stats=10):
        #setting params
        self.weights = G1DList.G1DList(self.n_input)
        lim = np.sqrt(6)/np.sqrt((self.fan_in+self.fan_out))
        #Khoi tao trong so
        self.weights.setParams(rangemin=-lim,rangemax=lim)
        # cai dat ham khoi tao
        self.weights.initializator.set(Initializators.G1DListInitializatorReal)
        #cai dat ham dot bien
        self.weights.mutator.set(Mutators.G1DListMutatorRealGaussian)
        #cai dat ham do thich nghi
        self.weights.evaluator.set(self.eval_score)
        # cai dat ham lai ghep
        self.weights.crossover.set(Crossovers.G1DListCrossoverUniform)
        #code genetic
        self.ga = GSimpleGA.GSimpleGA(self.weights)
        self.ga.selector.set(Selectors.GRouletteWheel)
        self.ga.setGenerations(400)
        self.ga.evolve(freq_stats=10)
        return self.score(self.X_data,self.y_data)
    def getParam(self):
        return np.array(self.ga.bestIndividual().genomeList)
    def predict(self,X):
        return np.dot(X,np.array(self.ga.bestIndividual().genomeList))
    def score(self,X,y_actual):
        return mean_squared_error(y_actual,self.predict(X))



