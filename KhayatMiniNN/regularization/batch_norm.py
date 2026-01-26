import numpy as np
from .base import RegularizationLayer


class BatchNormalization(RegularizationLayer):
    
    def __init__(self, num_features, momentum=0.9, epsilon=1e-5, name="BatchNorm"):
        super().__init__(name)
        self.num_features = num_features
        self.momentum = momentum
        self.epsilon = epsilon
        self.training = True  # Ensure training flag is set
        
        self.params = {
            'gamma': np.ones(num_features),
            'beta': np.zeros(num_features)
        }
        
        self.grads = {
            'gamma': np.zeros(num_features),
            'beta': np.zeros(num_features)
        }
        
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        
        self.cache = {}
    
    def forward(self, input_data):
        if self.training:
            batch_mean = np.mean(input_data, axis=0, keepdims=True)
            batch_var = np.var(input_data, axis=0, keepdims=True)
            
            input_normalized = (input_data - batch_mean) / np.sqrt(batch_var + self.epsilon)
            
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean.squeeze()
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var.squeeze()
            
            self.cache['input'] = input_data
            self.cache['normalized'] = input_normalized
            self.cache['mean'] = batch_mean
            self.cache['var'] = batch_var
        else:
            # Use running statistics during inference
            if self.running_mean.ndim == 0:
                running_mean = self.running_mean
                running_var = self.running_var
            else:
                running_mean = self.running_mean.reshape(1, -1) if len(self.running_mean.shape) == 1 else self.running_mean
                running_var = self.running_var.reshape(1, -1) if len(self.running_var.shape) == 1 else self.running_var
            input_normalized = (input_data - running_mean) / np.sqrt(running_var + self.epsilon)
        
        gamma = self.params['gamma']
        beta = self.params['beta']
        if gamma.ndim == 1:
            gamma = gamma.reshape(1, -1)
        if beta.ndim == 1:
            beta = beta.reshape(1, -1)
        
        output = gamma * input_normalized + beta
        return output
    
    def backward(self, output_grad):
        if not self.training or 'normalized' not in self.cache:
            return output_grad
        
        batch_size = output_grad.shape[0]
        
        normalized = self.cache['normalized']
        self.grads['gamma'] = np.sum(output_grad * normalized, axis=0) / batch_size
        self.grads['beta'] = np.sum(output_grad, axis=0) / batch_size
        
        gamma = self.params['gamma']
        if gamma.ndim == 1:
            gamma = gamma.reshape(1, -1)
        grad_normalized = output_grad * gamma
        
        var = self.cache['var']
        mean = self.cache['mean']
        input_data = self.cache['input']
        
        grad_var = np.sum(grad_normalized * (input_data - mean) * -0.5 * (var + self.epsilon) ** -1.5, axis=0, keepdims=True)
        grad_mean = (np.sum(grad_normalized * -1 / np.sqrt(var + self.epsilon), axis=0, keepdims=True) +
                     grad_var * np.mean(-2 * (input_data - mean), axis=0, keepdims=True))
        
        grad_input = (grad_normalized / np.sqrt(var + self.epsilon) +
                      grad_var * 2 * (input_data - mean) / batch_size +
                      grad_mean / batch_size)
        
        return grad_input
    
    def get_params(self):
        return self.params
    
    def get_grads(self):
        return self.grads
    
    def set_params(self, params):
        if 'gamma' in params:
            self.params['gamma'] = params['gamma']
        if 'beta' in params:
            self.params['beta'] = params['beta']
