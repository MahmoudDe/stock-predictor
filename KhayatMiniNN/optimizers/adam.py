import numpy as np
from .base import Optimizer
from ..gpu_utils import get_xp


class Adam(Optimizer):
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0, name="Adam", device_manager=None):
        """
        Adam optimizer with optional weight decay (L2 regularization).
        
        Args:
            learning_rate: Learning rate for parameter updates
            beta1: Exponential decay rate for first moment estimates
            beta2: Exponential decay rate for second moment estimates
            epsilon: Small constant for numerical stability
            weight_decay: L2 regularization coefficient (default: 0.0, no regularization)
            name: Name of the optimizer
            device_manager: Optional device manager for GPU support
        """
        super().__init__(learning_rate, name)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.t = 0
        self.m = {}
        self.v = {}
        self.device_manager = device_manager
        self.xp = get_xp(device_manager) if device_manager else np
    
    def update(self, params, grads):
        if not self.m:
            for param_name in params:
                self.m[param_name] = self.xp.zeros_like(params[param_name])
                self.v[param_name] = self.xp.zeros_like(params[param_name])
        
        self.t += 1
        updated_params = {}
        
        for param_name in params:
            if param_name in grads:
                # Apply weight decay (L2 regularization) to gradients
                # Only apply to weight matrices, not biases (heuristic: bias params usually have smaller shape)
                grad = grads[param_name]
                if self.weight_decay > 0 and params[param_name].ndim >= 2:
                    grad = grad + self.weight_decay * params[param_name]
                
                self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * grad
                self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * (grad ** 2)
                
                m_hat = self.m[param_name] / (1 - self.beta1 ** self.t)
                v_hat = self.v[param_name] / (1 - self.beta2 ** self.t)
                
                updated_params[param_name] = params[param_name] - self.learning_rate * m_hat / (self.xp.sqrt(v_hat) + self.epsilon)
            else:
                updated_params[param_name] = params[param_name]
        
        return updated_params
