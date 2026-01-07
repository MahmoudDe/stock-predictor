from abc import ABC, abstractmethod


class RegularizationLayer(ABC):
    
    def __init__(self, name="RegularizationLayer"):
        self.name = name
        self.training = True
    
    @abstractmethod
    def forward(self, input_data):
        pass
    
    @abstractmethod
    def backward(self, output_grad):
        pass
    
    def set_training(self, training=True):
        self.training = training
