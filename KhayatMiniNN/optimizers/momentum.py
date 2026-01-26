import numpy as np
from .base import Optimizer
from ..gpu_utils import get_xp


class Momentum(Optimizer):
    
    def __init__(self, learning_rate=0.01, momentum=0.9, name="Momentum", device_manager=None):
        super().__init__(learning_rate, name)
        self.momentum = momentum
        self.velocity = {}
        self.device_manager = device_manager
        self.xp = get_xp(device_manager) if device_manager else np
    
    def update(self, params, grads):
        if not self.velocity:
            for param_name in params:
                self.velocity[param_name] = self.xp.zeros_like(params[param_name])
        
        updated_params = {}
        
        for param_name in params:
            if param_name in grads:
                self.velocity[param_name] = (
                    self.momentum * self.velocity[param_name] - 
                    self.learning_rate * grads[param_name]
                )
                updated_params[param_name] = params[param_name] + self.velocity[param_name]
            else:
                updated_params[param_name] = params[param_name]
        
        return updated_params
