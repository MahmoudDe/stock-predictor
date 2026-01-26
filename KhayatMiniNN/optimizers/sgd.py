import numpy as np
from .base import Optimizer
from ..gpu_utils import get_xp


class SGD(Optimizer):
    
    def __init__(self, learning_rate=0.01, name="SGD", device_manager=None):
        super().__init__(learning_rate, name)
        self.device_manager = device_manager
        self.xp = get_xp(device_manager) if device_manager else np
    
    def update(self, params, grads):
        updated_params = {}
        
        for param_name in params:
            if param_name in grads:
                updated_params[param_name] = params[param_name] - self.learning_rate * grads[param_name]
            else:
                updated_params[param_name] = params[param_name]
        
        return updated_params
