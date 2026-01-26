import numpy as np
from .base import Loss
from ..utils import softmax
from ..gpu_utils import get_xp


class SoftmaxCrossEntropy(Loss):
    
    def __init__(self, name="SoftmaxCrossEntropy", device_manager=None):
        super().__init__(name)
        self.device_manager = device_manager
        self.xp = get_xp(device_manager) if device_manager else np
    
    def forward(self, predictions, targets):
        self.output = predictions
        self.target = targets
        
        # Use GPU-aware softmax if available
        if self.device_manager and self.device_manager.use_gpu:
            x_max = self.xp.max(predictions, axis=-1, keepdims=True)
            x_shifted = predictions - x_max
            exp_x = self.xp.exp(x_shifted)
            self.softmax_output = exp_x / self.xp.sum(exp_x, axis=-1, keepdims=True)
        else:
            self.softmax_output = softmax(predictions)
        
        batch_size = predictions.shape[0]
        
        eps = 1e-8
        softmax_clipped = self.xp.clip(self.softmax_output, eps, 1.0)
        loss = -self.xp.sum(targets * self.xp.log(softmax_clipped)) / batch_size
        
        return loss
    
    def backward(self):
        batch_size = self.output.shape[0]
        gradient = (self.softmax_output - self.target) / batch_size
        return gradient
