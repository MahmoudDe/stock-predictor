import numpy as np
from .base import Layer
from ..utils import initialize_weights


class Dense(Layer):
    
    def __init__(self, input_size, output_size, name="Dense"):
        super().__init__(name)
        self.input_size = input_size
        self.output_size = output_size
        
        self.params['W'] = initialize_weights((input_size, output_size), method='xavier')
        self.params['b'] = np.zeros((1, output_size))
        
        self.grads['W'] = np.zeros_like(self.params['W'])
        self.grads['b'] = np.zeros_like(self.params['b'])
    
    def forward(self, input_data):
        self.input = input_data
        output = np.dot(input_data, self.params['W']) + self.params['b']
        return output
    
    def backward(self, output_grad):
        batch_size = self.input.shape[0]
        
        self.grads['W'] = np.dot(self.input.T, output_grad) / batch_size
        self.grads['b'] = np.sum(output_grad, axis=0, keepdims=True) / batch_size
        
        input_grad = np.dot(output_grad, self.params['W'].T)
        return input_grad
