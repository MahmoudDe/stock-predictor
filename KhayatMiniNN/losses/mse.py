import numpy as np
from .base import Loss
from ..gpu_utils import get_xp


class MeanSquaredError(Loss):
    
    def __init__(self, name="MSE", device_manager=None):
        super().__init__(name)
        self.device_manager = device_manager
        self.xp = get_xp(device_manager) if device_manager else np
    
    def forward(self, predictions, targets):
        self.output = predictions
        self.target = targets
        squared_errors = (predictions - targets) ** 2
        loss = self.xp.mean(squared_errors)
        return loss
    
    def backward(self):
        batch_size = self.output.shape[0]
        gradient = 2 * (self.output - self.target) / batch_size
        return gradient
