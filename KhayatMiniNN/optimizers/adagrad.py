import numpy as np
from .base import Optimizer
from ..gpu_utils import get_xp


class Adagrad(Optimizer):
    
    def __init__(self, learning_rate=0.01, epsilon=1e-8, name="Adagrad", device_manager=None):
        super().__init__(learning_rate, name)
        self.epsilon = epsilon
        self.cache = {}
        self.device_manager = device_manager
        self.xp = get_xp(device_manager) if device_manager else np
    
    def update(self, params, grads):
        if not self.cache:
            for param_name in params:
                self.cache[param_name] = self.xp.zeros_like(params[param_name])
        
        updated_params = {}
        
        for param_name in params:
            if param_name in grads:
                self.cache[param_name] += grads[param_name] ** 2
                adjusted_lr = self.learning_rate / (self.xp.sqrt(self.cache[param_name]) + self.epsilon)
                updated_params[param_name] = params[param_name] - adjusted_lr * grads[param_name]
            else:
                updated_params[param_name] = params[param_name]
        
        return updated_params
