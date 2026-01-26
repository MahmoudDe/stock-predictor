import numpy as np
from .base import Layer
from ..utils import initialize_weights


class Embedding(Layer):
    """
    Embedding layer for converting categorical indices to dense vectors.
    
    Useful for encoding stock tickers or other categorical features.
    """
    
    def __init__(self, vocab_size, embedding_dim, name="Embedding"):
        """
        Args:
            vocab_size: Size of vocabulary (number of unique categories)
            embedding_dim: Dimension of embedding vectors
        """
        super().__init__(name)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Initialize embedding matrix: (vocab_size, embedding_dim)
        self.params['W'] = initialize_weights(
            (vocab_size, embedding_dim), 
            method='xavier'
        )
        
        # Initialize gradients
        self.grads['W'] = np.zeros_like(self.params['W'])
        
        # Cache for backward pass
        self.cache = {}
    
    def forward(self, input_data):
        """
        Args:
            input_data: (batch_size, sequence_length) or (batch_size,) - integer indices
        
        Returns:
            output: (batch_size, sequence_length, embedding_dim) or (batch_size, embedding_dim)
        """
        self.input = input_data
        
        # Ensure input is integer type
        if input_data.dtype != np.int32 and input_data.dtype != np.int64:
            input_data = input_data.astype(np.int32)
        
        # Handle both 1D and 2D inputs
        if input_data.ndim == 1:
            input_data = input_data.reshape(-1, 1)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, seq_len = input_data.shape
        
        # Lookup embeddings: (batch_size, seq_len, embedding_dim)
        output = self.params['W'][input_data]
        
        if squeeze_output:
            output = output.squeeze(1)  # (batch_size, embedding_dim)
        
        # Store for backward pass
        self.cache = {
            'input_shape': input_data.shape,
            'squeeze_output': squeeze_output
        }
        
        return output
    
    def backward(self, output_grad):
        """
        Args:
            output_grad: (batch_size, sequence_length, embedding_dim) or (batch_size, embedding_dim)
        
        Returns:
            input_grad: None (indices are not differentiable, but we update embedding matrix)
        """
        input_data = self.input
        
        # Handle both 1D and 2D inputs
        was_1d = False
        if input_data.ndim == 1:
            was_1d = True
            input_data = input_data.reshape(-1, 1)
            if output_grad.ndim == 2:
                output_grad = output_grad.reshape(-1, 1, output_grad.shape[-1])
        
        batch_size, seq_len = input_data.shape
        
        # Ensure output_grad is 3D
        if output_grad.ndim == 2:
            output_grad = output_grad.reshape(batch_size, 1, output_grad.shape[-1])
        
        # Reset gradient
        self.grads['W'] = np.zeros_like(self.params['W'])
        
        # Accumulate gradients for each unique index
        unique_indices = np.unique(input_data)
        
        for idx in unique_indices:
            # Find all positions where this index appears
            # mask shape: (batch_size, seq_len)
            mask = (input_data == idx)
            
            # Sum gradients for this index across all positions
            # output_grad shape: (batch_size, seq_len, embedding_dim)
            # We want to sum over all positions where mask is True
            # Expand mask to match output_grad dimensions
            mask_expanded = np.expand_dims(mask, axis=2)  # (batch_size, seq_len, 1)
            mask_expanded = np.broadcast_to(mask_expanded, output_grad.shape)  # (batch_size, seq_len, embedding_dim)
            
            # Sum gradients where mask is True
            grad_sum = np.sum(output_grad * mask_expanded, axis=(0, 1))  # (embedding_dim,)
            
            # Accumulate gradient
            self.grads['W'][idx] += grad_sum / batch_size
        
        # Indices are not differentiable, return None
        # (In practice, we don't need to return gradient for indices)
        return None

