from abc import ABC, abstractmethod


class Optimizer(ABC):
    
    def __init__(self, learning_rate=0.01, name="Optimizer"):
        self.learning_rate = learning_rate
        self.name = name
    
    @abstractmethod
    def update(self, params, grads):
        pass
