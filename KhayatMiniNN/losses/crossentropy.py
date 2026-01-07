import numpy as np
from .base import Loss
from ..utils import softmax


class SoftmaxCrossEntropy(Loss):
    
    def __init__(self, name="SoftmaxCrossEntropy"):
        super().__init__(name)
    
    def forward(self, predictions, targets):
        self.output = predictions
        self.target = targets
        
        self.softmax_output = softmax(predictions)
        batch_size = predictions.shape[0]
        
        eps = 1e-8
        softmax_clipped = np.clip(self.softmax_output, eps, 1.0)
        loss = -np.sum(targets * np.log(softmax_clipped)) / batch_size
        
        return loss
    
    def backward(self):
        batch_size = self.output.shape[0]
        gradient = (self.softmax_output - self.target) / batch_size
        return gradient
