import numpy as np
from .base import RegularizationLayer


class Dropout(RegularizationLayer):
    
    def __init__(self, dropout_rate=0.5, name="Dropout"):
        super().__init__(name)
        self.dropout_rate = dropout_rate
        self.training = True
        self.mask = None
    
    def forward(self, input_data):
        self.input = input_data
        
        if self.training and self.dropout_rate > 0:
            keep_prob = 1 - self.dropout_rate
            self.mask = np.random.binomial(1, keep_prob, input_data.shape)
            output = input_data * self.mask / keep_prob
        else:
            output = input_data
            self.mask = np.ones_like(input_data)
        
        return output
    
    def backward(self, output_grad):
        if self.training and self.dropout_rate > 0:
            keep_prob = 1 - self.dropout_rate
            return output_grad * self.mask / keep_prob
        else:
            return output_grad
    
    def set_training(self, training=True):
        self.training = training
