import numpy as np
from .base import Layer


class Linear(Layer):
    
    def __init__(self, name="Linear"):
        super().__init__(name)
    
    def forward(self, input_data):
        self.input = input_data
        return input_data
    
    def backward(self, output_grad):
        return output_grad


class ReLU(Layer):
    
    def __init__(self, name="ReLU"):
        super().__init__(name)
    
    def forward(self, input_data):
        self.input = input_data
        self.output = np.maximum(0, input_data)
        return self.output
    
    def backward(self, output_grad):
        mask = (self.input > 0).astype(float)
        return output_grad * mask


class Sigmoid(Layer):
    
    def __init__(self, name="Sigmoid"):
        super().__init__(name)
    
    def forward(self, input_data):
        self.input = input_data
        self.output = 1 / (1 + np.exp(-np.clip(input_data, -500, 500)))
        return self.output
    
    def backward(self, output_grad):
        return output_grad * self.output * (1 - self.output)


class Tanh(Layer):
    
    def __init__(self, name="Tanh"):
        super().__init__(name)
    
    def forward(self, input_data):
        self.input = input_data
        self.output = np.tanh(input_data)
        return self.output
    
    def backward(self, output_grad):
        return output_grad * (1 - self.output ** 2)
