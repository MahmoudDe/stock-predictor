import numpy as np
from .base import Loss


class MeanSquaredError(Loss):
    
    def __init__(self, name="MSE"):
        super().__init__(name)
    
    def forward(self, predictions, targets):
        self.output = predictions
        self.target = targets
        squared_errors = (predictions - targets) ** 2
        loss = np.mean(squared_errors)
        return loss
    
    def backward(self):
        batch_size = self.output.shape[0]
        gradient = 2 * (self.output - self.target) / batch_size
        return gradient
