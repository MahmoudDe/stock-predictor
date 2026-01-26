import numpy as np
from .base import Layer


class MaxPooling1D(Layer):
    """
    1D Max Pooling layer.
    
    Reduces sequence length by taking maximum over pooling windows.
    """
    
    def __init__(self, pool_size=2, stride=None, name="MaxPooling1D"):
        """
        Args:
            pool_size: Size of the pooling window
            stride: Stride of pooling (default: same as pool_size)
        """
        super().__init__(name)
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size
        self.cache = {}
    
    def forward(self, input_data):
        """
        Args:
            input_data: (batch_size, sequence_length, channels)
        
        Returns:
            output: (batch_size, output_length, channels)
        """
        self.input = input_data
        batch_size, seq_len, channels = input_data.shape
        
        # Compute output length
        output_length = (seq_len - self.pool_size) // self.stride + 1
        
        # Initialize output
        output = np.zeros((batch_size, output_length, channels))
        max_indices = np.zeros((batch_size, output_length, channels), dtype=int)
        
        # Perform max pooling
        for i in range(output_length):
            start_idx = i * self.stride
            end_idx = start_idx + self.pool_size
            
            # Extract window: (batch_size, pool_size, channels)
            window = input_data[:, start_idx:end_idx, :]
            
            # Find max along pool_size dimension
            output[:, i, :] = np.max(window, axis=1)
            
            # Store indices for backward pass
            max_indices[:, i, :] = start_idx + np.argmax(window, axis=1)
        
        # Store for backward pass
        self.cache = {
            'max_indices': max_indices,
            'output_length': output_length
        }
        
        return output
    
    def backward(self, output_grad):
        """
        Args:
            output_grad: (batch_size, output_length, channels)
        
        Returns:
            input_grad: (batch_size, sequence_length, channels)
        """
        batch_size, seq_len, channels = self.input.shape
        output_length = self.cache['output_length']
        max_indices = self.cache['max_indices']
        
        # Initialize gradient
        input_grad = np.zeros_like(self.input)
        
        # Route gradients to max positions
        for i in range(output_length):
            for b in range(batch_size):
                for c in range(channels):
                    idx = max_indices[b, i, c]
                    input_grad[b, idx, c] += output_grad[b, i, c]
        
        return input_grad


class AveragePooling1D(Layer):
    """
    1D Average Pooling layer.
    
    Reduces sequence length by taking average over pooling windows.
    """
    
    def __init__(self, pool_size=2, stride=None, name="AveragePooling1D"):
        """
        Args:
            pool_size: Size of the pooling window
            stride: Stride of pooling (default: same as pool_size)
        """
        super().__init__(name)
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size
    
    def forward(self, input_data):
        """
        Args:
            input_data: (batch_size, sequence_length, channels)
        
        Returns:
            output: (batch_size, output_length, channels)
        """
        self.input = input_data
        batch_size, seq_len, channels = input_data.shape
        
        # Compute output length
        output_length = (seq_len - self.pool_size) // self.stride + 1
        
        # Initialize output
        output = np.zeros((batch_size, output_length, channels))
        
        # Perform average pooling
        for i in range(output_length):
            start_idx = i * self.stride
            end_idx = start_idx + self.pool_size
            
            # Extract window: (batch_size, pool_size, channels)
            window = input_data[:, start_idx:end_idx, :]
            
            # Average along pool_size dimension
            output[:, i, :] = np.mean(window, axis=1)
        
        return output
    
    def backward(self, output_grad):
        """
        Args:
            output_grad: (batch_size, output_length, channels)
        
        Returns:
            input_grad: (batch_size, sequence_length, channels)
        """
        batch_size, seq_len, channels = self.input.shape
        output_length = output_grad.shape[1]
        
        # Initialize gradient
        input_grad = np.zeros_like(self.input)
        
        # Distribute gradients evenly
        for i in range(output_length):
            start_idx = i * self.stride
            end_idx = start_idx + self.pool_size
            
            # Distribute gradient equally to all positions in window
            grad_per_position = output_grad[:, i:i+1, :] / self.pool_size
            input_grad[:, start_idx:end_idx, :] += grad_per_position
        
        return input_grad


