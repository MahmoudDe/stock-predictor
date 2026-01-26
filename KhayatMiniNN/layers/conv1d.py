import numpy as np
from .base import Layer
from ..utils import initialize_weights


class Conv1D(Layer):
    """
    1D Convolution layer for sequential data.
    
    Processes sequences of shape (batch_size, sequence_length, input_channels)
    and outputs (batch_size, output_length, output_channels).
    """
    
    def __init__(self, input_channels, output_channels, kernel_size, 
                 stride=1, padding='valid', name="Conv1D"):
        """
        Args:
            input_channels: Number of input channels/features
            output_channels: Number of output channels/filters
            kernel_size: Size of the convolution kernel
            stride: Stride of the convolution (default: 1)
            padding: 'valid' (no padding) or 'same' (pad to maintain size)
        """
        super().__init__(name)
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights and biases
        # Filters shape: (kernel_size, input_channels, output_channels)
        self.params['W'] = initialize_weights(
            (kernel_size, input_channels, output_channels), 
            method='he'
        )
        self.params['b'] = np.zeros((1, output_channels))
        
        # Initialize gradients
        self.grads['W'] = np.zeros_like(self.params['W'])
        self.grads['b'] = np.zeros_like(self.params['b'])
        
        # Cache for backward pass
        self.cache = {}
    
    def _compute_padding(self, input_length):
        """Compute padding size for 'same' padding."""
        if self.padding == 'same':
            # For 'same' padding, output_length = input_length
            # output_length = (input_length + 2*pad - kernel_size) // stride + 1
            # Solving for pad:
            pad = ((input_length - 1) * self.stride + self.kernel_size - input_length) // 2
            return pad
        elif self.padding == 'valid':
            return 0
        else:
            raise ValueError(f"Unknown padding mode: {self.padding}")
    
    def _pad_input(self, x, pad):
        """Pad input sequence."""
        if pad == 0:
            return x
        # Pad along sequence dimension (axis=1)
        return np.pad(x, ((0, 0), (pad, pad), (0, 0)), mode='constant', constant_values=0)
    
    def forward(self, input_data):
        """
        Args:
            input_data: (batch_size, sequence_length, input_channels)
        
        Returns:
            output: (batch_size, output_length, output_channels)
        """
        self.input = input_data
        batch_size, seq_len, input_channels = input_data.shape
        
        # Compute padding
        pad = self._compute_padding(seq_len)
        x_padded = self._pad_input(input_data, pad)
        
        # Compute output length
        output_length = (seq_len + 2 * pad - self.kernel_size) // self.stride + 1
        
        # Initialize output
        output = np.zeros((batch_size, output_length, self.output_channels))
        
        # Perform convolution
        for i in range(output_length):
            start_idx = i * self.stride
            end_idx = start_idx + self.kernel_size
            
            # Extract window: (batch_size, kernel_size, input_channels)
            window = x_padded[:, start_idx:end_idx, :]
            
            # Reshape for matrix multiplication: (batch_size, kernel_size * input_channels)
            window_flat = window.reshape(batch_size, -1)
            
            # Reshape weights: (kernel_size * input_channels, output_channels)
            W_flat = self.params['W'].reshape(-1, self.output_channels)
            
            # Convolution: (batch_size, output_channels)
            output[:, i, :] = np.dot(window_flat, W_flat) + self.params['b']
        
        # Store for backward pass
        self.cache = {
            'x_padded': x_padded,
            'pad': pad,
            'output_length': output_length
        }
        
        return output
    
    def backward(self, output_grad):
        """
        Args:
            output_grad: (batch_size, output_length, output_channels)
        
        Returns:
            input_grad: (batch_size, sequence_length, input_channels)
        """
        batch_size, output_length, output_channels = output_grad.shape
        seq_len = self.input.shape[1]
        
        x_padded = self.cache['x_padded']
        pad = self.cache['pad']
        
        # Initialize gradients
        input_grad_padded = np.zeros_like(x_padded)
        self.grads['W'] = np.zeros_like(self.params['W'])
        self.grads['b'] = np.zeros((1, output_channels))
        
        # Backward pass
        for i in range(output_length):
            start_idx = i * self.stride
            end_idx = start_idx + self.kernel_size
            
            # Extract window
            window = x_padded[:, start_idx:end_idx, :]
            window_flat = window.reshape(batch_size, -1)
            
            # Gradient w.r.t. output
            grad_out = output_grad[:, i, :]  # (batch_size, output_channels)
            
            # Gradient w.r.t. weights
            W_flat = self.params['W'].reshape(-1, output_channels)
            self.grads['W'] += np.dot(
                window_flat.T, grad_out
            ).reshape(self.kernel_size, self.input_channels, output_channels) / batch_size
            
            # Gradient w.r.t. bias
            self.grads['b'] += np.sum(grad_out, axis=0, keepdims=True) / batch_size
            
            # Gradient w.r.t. input (unfolded)
            grad_window_flat = np.dot(grad_out, W_flat.T)  # (batch_size, kernel_size * input_channels)
            grad_window = grad_window_flat.reshape(batch_size, self.kernel_size, self.input_channels)
            
            # Accumulate gradients
            input_grad_padded[:, start_idx:end_idx, :] += grad_window
        
        # Remove padding
        if pad > 0:
            input_grad = input_grad_padded[:, pad:-pad, :]
        else:
            input_grad = input_grad_padded
        
        return input_grad


