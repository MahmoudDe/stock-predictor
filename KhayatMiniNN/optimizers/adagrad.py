import numpy as np
from .base import Optimizer


class Adagrad(Optimizer):
    
    def __init__(self, learning_rate=0.01, epsilon=1e-8, name="Adagrad"):
        super().__init__(learning_rate, name)
        self.epsilon = epsilon
        self.cache = {}
    
    def update(self, params, grads):
        if not self.cache:
            for param_name in params:
                self.cache[param_name] = np.zeros_like(params[param_name])
        
        updated_params = {}
        
        for param_name in params:
            if param_name in grads:
                self.cache[param_name] += grads[param_name] ** 2
                adjusted_lr = self.learning_rate / (np.sqrt(self.cache[param_name]) + self.epsilon)
                updated_params[param_name] = params[param_name] - adjusted_lr * grads[param_name]
            else:
                updated_params[param_name] = params[param_name]
        
        return updated_params
