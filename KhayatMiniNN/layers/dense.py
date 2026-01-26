import numpy as np
from .base import Layer
from ..utils import initialize_weights
from ..gpu_utils import get_xp, to_device


class Dense(Layer):
    
    def __init__(self, input_size, output_size, name="Dense", device_manager=None):
        super().__init__(name)
        self.input_size = input_size
        self.output_size = output_size
        self.device_manager = device_manager
        self.xp = get_xp(device_manager) if device_manager else np
        
        # Initialize weights on device
        weights = initialize_weights((input_size, output_size), method='xavier')
        self.params['W'] = to_device(weights, device_manager) if device_manager else weights
        self.params['b'] = self.xp.zeros((1, output_size))
        
        self.grads['W'] = self.xp.zeros_like(self.params['W'])
        self.grads['b'] = self.xp.zeros_like(self.params['b'])
    
    def forward(self, input_data):
        self.input = input_data
        output = self.xp.dot(input_data, self.params['W']) + self.params['b']
        return output
    
    def backward(self, output_grad):
        batch_size = self.input.shape[0]
        
        self.grads['W'] = self.xp.dot(self.input.T, output_grad) / batch_size
        self.grads['b'] = self.xp.sum(output_grad, axis=0, keepdims=True) / batch_size
        
        input_grad = self.xp.dot(output_grad, self.params['W'].T)
        return input_grad
