import numpy as np
from .base import Loss
from ..gpu_utils import get_xp


class BinaryCrossEntropy(Loss):
    """
    Binary Cross-Entropy Loss for binary classification.
    
    Can work with:
    - Raw logits (recommended): predictions are raw scores, targets are 0/1
    - Sigmoid outputs: predictions are already sigmoided (0-1), targets are 0/1
    """
    
    def __init__(self, from_logits=True, name="BinaryCrossEntropy", device_manager=None):
        """
        Args:
            from_logits: If True, predictions are raw logits (will apply sigmoid).
                        If False, predictions are already in [0,1] range.
            device_manager: DeviceManager instance for GPU support
        """
        super().__init__(name)
        self.from_logits = from_logits
        self.device_manager = device_manager
        self.xp = get_xp(device_manager) if device_manager else np
    
    def _sigmoid(self, x):
        """Numerically stable sigmoid."""
        x_clipped = self.xp.clip(x, -500, 500)
        return 1 / (1 + self.xp.exp(-x_clipped))
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: (batch_size, 1) or (batch_size,) - predicted scores
            targets: (batch_size, 1) or (batch_size,) - true labels (0 or 1)
        
        Returns:
            loss: scalar binary cross-entropy loss
        """
        self.output = predictions
        self.target = targets
        
        # Ensure targets are in correct shape
        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        
        batch_size = predictions.shape[0]
        
        # Apply sigmoid if from_logits
        if self.from_logits:
            self.sigmoid_output = self._sigmoid(predictions)
        else:
            # Clip to avoid numerical issues
            self.sigmoid_output = self.xp.clip(predictions, 1e-8, 1 - 1e-8)
        
        # Compute binary cross-entropy
        eps = 1e-8
        sigmoid_clipped = self.xp.clip(self.sigmoid_output, eps, 1 - eps)
        
        loss = -self.xp.mean(
            targets * self.xp.log(sigmoid_clipped) + 
            (1 - targets) * self.xp.log(1 - sigmoid_clipped)
        )
        
        return loss
    
    def backward(self):
        """
        Returns:
            gradient: gradient w.r.t. predictions
        """
        batch_size = self.output.shape[0]
        
        if self.from_logits:
            # Gradient for sigmoid + BCE
            gradient = (self.sigmoid_output - self.target) / batch_size
        else:
            # Gradient when predictions are already sigmoided
            eps = 1e-8
            sigmoid_clipped = self.xp.clip(self.sigmoid_output, eps, 1 - eps)
            gradient = -(self.target / sigmoid_clipped - 
                        (1 - self.target) / (1 - sigmoid_clipped)) / batch_size
        
        return gradient


