from sklearn.base import BaseEstimator
import numpy as np
class ACONNEstimator(BaseEstimator):
    def get_params(self, deep=True):
        return {
            "epsilon":self.epsilon,
            "max_epochs":self.max_epochs,
            "number_of_solutions":self.number_of_solutions,
            "Q":self.Q,
            "error_criteria":self.error_criteria,
            "number_of_weights":self.number_of_weights,
            "neural_shape":self.neural_shape,
            "const_sd":self.const_sd
        }
    def set_params(self, **params):
        for param,value in params.items():
            self.__setattr__(param,value)
        return self