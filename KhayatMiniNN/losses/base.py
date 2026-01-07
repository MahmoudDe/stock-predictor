from abc import ABC, abstractmethod


class Loss(ABC):
    
    def __init__(self, name="Loss"):
        self.name = name
        self.output = None
        self.target = None
    
    @abstractmethod
    def forward(self, predictions, targets):
        pass
    
    @abstractmethod
    def backward(self):
        pass
