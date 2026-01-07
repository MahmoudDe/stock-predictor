import numpy as np
from .base import Optimizer


class Adam(Optimizer):
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, name="Adam"):
        super().__init__(learning_rate, name)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = {}
        self.v = {}
    
    def update(self, params, grads):
        if not self.m:
            for param_name in params:
                self.m[param_name] = np.zeros_like(params[param_name])
                self.v[param_name] = np.zeros_like(params[param_name])
        
        self.t += 1
        updated_params = {}
        
        for param_name in params:
            if param_name in grads:
                self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * grads[param_name]
                self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * (grads[param_name] ** 2)
                
                m_hat = self.m[param_name] / (1 - self.beta1 ** self.t)
                v_hat = self.v[param_name] / (1 - self.beta2 ** self.t)
                
                updated_params[param_name] = params[param_name] - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            else:
                updated_params[param_name] = params[param_name]
        
        return updated_params
