from abc import ABC, abstractmethod


class Layer(ABC):
    
    def __init__(self, name="Layer"):
        self.name = name
        self.input = None
        self.params = {}
        self.grads = {}
    
    @abstractmethod
    def forward(self, input_data):
        pass
    
    @abstractmethod
    def backward(self, output_grad):
        pass
    
    def get_params(self):
        return self.params
    
    def get_grads(self):
        return self.grads
    
    def set_params(self, params):
        self.params = params
