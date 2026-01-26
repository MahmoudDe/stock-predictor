import numpy as np
from .base import Layer
from ..gpu_utils import get_xp


class Linear(Layer):
    
    def __init__(self, name="Linear", device_manager=None):
        super().__init__(name)
        self.device_manager = device_manager
        self.xp = get_xp(device_manager) if device_manager else np
    
    def forward(self, input_data):
        self.input = input_data
        return input_data
    
    def backward(self, output_grad):
        return output_grad


class ReLU(Layer):
    
    def __init__(self, name="ReLU", device_manager=None):
        super().__init__(name)
        self.device_manager = device_manager
        self.xp = get_xp(device_manager) if device_manager else np
    
    def forward(self, input_data):
        self.input = input_data
        self.output = self.xp.maximum(0, input_data)
        return self.output
    
    def backward(self, output_grad):
        mask = (self.input > 0).astype(float)
        return output_grad * mask


class Sigmoid(Layer):
    
    def __init__(self, name="Sigmoid", device_manager=None):
        super().__init__(name)
        self.device_manager = device_manager
        self.xp = get_xp(device_manager) if device_manager else np
    
    def forward(self, input_data):
        self.input = input_data
        self.output = 1 / (1 + self.xp.exp(-self.xp.clip(input_data, -500, 500)))
        return self.output
    
    def backward(self, output_grad):
        return output_grad * self.output * (1 - self.output)


class Tanh(Layer):
    
    def __init__(self, name="Tanh", device_manager=None):
        super().__init__(name)
        self.device_manager = device_manager
        self.xp = get_xp(device_manager) if device_manager else np
    
    def forward(self, input_data):
        self.input = input_data
        self.output = self.xp.tanh(input_data)
        return self.output
    
    def backward(self, output_grad):
        return output_grad * (1 - self.output ** 2)
