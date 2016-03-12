import numpy as np
from initializer import *
import logging
logging.basicConfig(level=logging.DEBUG,format='')
# In[150]:
class AntOptimization():
    def __init__(self,neural_shape,number_of_weights=None,number_of_solutions=100,max_epochs = 100,error_criteria = None,
                 epsilon = 0.75,evaporate = 0.08,const_sd = 0.1,**kwargs):
        self.epsilon = epsilon
        self.max_iteration = max_epochs
        self.error_criteria = error_criteria
        self.number_of_solutions = self.k = number_of_solutions
        self.number_of_weights = number_of_weights
        self.evaporate = evaporate
        self.const_sd = const_sd
        self.neural_shape = neural_shape
        self.Q = 0.08
        for key, value in kwargs.iteritems():      # styles is a regular dictionary
            print "%s=%s"%(key,value)

    def optimize(self,score_fn,summary_writter,X=None,y=None):
        self.feedFlow = score_fn
        self.summary_writter = summary_writter
        self.X = X
        self.y = y
        self.MAX_STAGMENT = 0.9
        self.best_loss = np.Inf
        self.best_archive = []
        archive = construct_solution(self.number_of_solutions,self.neural_shape)
        sorted_archive = self.calculate_fitness(self.feedFlow,archive,self.summary_writter)
        weights = self.calculate_weights(archive.shape)
        archive = self.sampling_more(sorted_archive,weights,self.evaporate)
        sorted_archive = self.calculate_fitness(self.feedFlow,archive,self.summary_writter)
        for i in np.arange(self.max_iteration):
            print "Epoch %d"%i
            # try:
                # print summary_writter[-1]
            if(sorted_archive==None):
                return archive[0]
            # best_error = summary_writter[-1]
            weights = self.calculate_weights(archive.shape)
            archive = self.sampling_more(sorted_archive,weights,self.epsilon)
            sorted_archive = self.calculate_fitness(self.feedFlow,archive,self.summary_writter)
            # except Exception as e:
            #     print e
            #     break
        print "Found best loss %f"%self.best_loss
        return self.best_archive

    def calculate_fitness(self,feedFlow,archive,summary_writter=None):
        fitness_solution = np.zeros(archive.shape[0])
        for (index,candidate) in enumerate(archive):
            feedFlow.set_weights(candidate)
            result = feedFlow.score(self.X,self.y)
            # print result
            fitness_solution[index] = result
        min_score = fitness_solution.min()
        # print min_score
        if(min_score >= self.MAX_STAGMENT):
            return None
        sorted_idx = np.argsort(fitness_solution)
        sorted_archive = np.zeros(archive.shape)
        for (index,item) in enumerate(sorted_idx):
            sorted_archive[item] = archive[index]
        if(min_score < self.best_loss):
            self.best_loss = min_score
            self.best_archive = sorted_archive[0]
        if(summary_writter!=None):
            summary_writter.append(fitness_solution.min())
        return sorted_archive
    # In[102]:

    def calculate_weights(self,shape):
        weights = np.zeros([shape[0],1])
        # qk = 0.1
        co_efficient = 1.0 / (0.1 * np.sqrt(2*np.pi))
        for index in np.arange(shape[0]):
            exponent = np.square(index-1) / (2*np.square(self.Q*shape[0]))
            weights[index] = np.multiply(co_efficient,np.exp(-exponent))
        return weights
    # In[103]:
    def compute_standard_deviation(self,i,l,archive,epsilon):
        __doc__ = " compute standard deviation with i, l, archive and epsilon"
        #Constant sd
        sd = 0.01
        sum_dev = np.abs(np.sum(archive[l]-archive[l][i])/(archive.shape[0]-1))
        # if(sum_dev <= 0.00001):
        #    return sd
        return np.multiply(sum_dev,epsilon)
    # In[104]:

    def choose_pdf(self,archive_shape,weights):
        sum_weights = np.sum(weights)
        temp = 0
        l = 0
        pro_r = np.random.uniform(0.0,1.0)
        for (index,weight) in enumerate(weights):
            temp = temp + weight/sum_weights
            if(temp > pro_r):
                l = index
        return l
    def sampling_more(self,archive,weights,epsilon):
        try:
            pdf = 0
            next_archive = np.zeros(archive.shape)
            for index in np.arange(archive.shape[0]):
                i_pdf = self.choose_pdf(archive.shape,weights)
                for item in np.arange(archive.shape[1]):
                    sigma = self.compute_standard_deviation(item,i_pdf,archive,epsilon)
                    mu = archive[pdf][item]
        #             print sigma,mu
                    next_archive[index][item] = np.random.normal(mu,sigma)
        except Exception as e:
            print "Sampling more"
            print e
        return next_archive
