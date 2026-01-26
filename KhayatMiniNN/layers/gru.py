import numpy as np
from .base import Layer
from ..utils import initialize_weights


class GRU(Layer):
    """
    Gated Recurrent Unit (GRU) layer for sequential data.
    
    Processes sequences of shape (batch_size, sequence_length, input_size)
    and outputs (batch_size, sequence_length, hidden_size) or 
    (batch_size, hidden_size) depending on return_sequences.
    """
    
    def __init__(self, input_size, hidden_size, return_sequences=False,
                 return_state=False, name="GRU"):
        """
        Args:
            input_size: Size of input features at each timestep
            hidden_size: Size of hidden state
            return_sequences: If True, return all timesteps. If False, return only last.
            return_state: If True, also return hidden state
        """
        super().__init__(name)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.return_sequences = return_sequences
        self.return_state = return_state
        
        # Initialize weights for reset gate, update gate, and candidate activation
        # W_z, W_r, W_h: (input_size, hidden_size) - for input
        # U_z, U_r, U_h: (hidden_size, hidden_size) - for hidden state
        # b_z, b_r, b_h: (hidden_size,) - biases
        
        # Update gate weights
        self.params['W_z'] = initialize_weights((input_size, hidden_size), method='xavier')
        self.params['U_z'] = initialize_weights((hidden_size, hidden_size), method='xavier')
        self.params['b_z'] = np.zeros((1, hidden_size))
        
        # Reset gate weights
        self.params['W_r'] = initialize_weights((input_size, hidden_size), method='xavier')
        self.params['U_r'] = initialize_weights((hidden_size, hidden_size), method='xavier')
        self.params['b_r'] = np.zeros((1, hidden_size))
        
        # Candidate activation weights
        self.params['W_h'] = initialize_weights((input_size, hidden_size), method='xavier')
        self.params['U_h'] = initialize_weights((hidden_size, hidden_size), method='xavier')
        self.params['b_h'] = np.zeros((1, hidden_size))
        
        # Initialize gradients
        for key in ['W_z', 'U_z', 'b_z', 'W_r', 'U_r', 'b_r', 
                   'W_h', 'U_h', 'b_h']:
            self.grads[key] = np.zeros_like(self.params[key])
        
        # Cache for backward pass
        self.cache = []
    
    def _sigmoid(self, x):
        """Sigmoid activation."""
        x_clipped = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_clipped))
    
    def _tanh(self, x):
        """Tanh activation."""
        return np.tanh(x)
    
    def forward(self, input_data):
        """
        Args:
            input_data: (batch_size, sequence_length, input_size)
        
        Returns:
            output: (batch_size, sequence_length, hidden_size) if return_sequences=True
                   (batch_size, hidden_size) if return_sequences=False
        """
        self.input = input_data
        batch_size, seq_len, _ = input_data.shape
        
        # Initialize hidden state
        h = np.zeros((batch_size, self.hidden_size))
        
        outputs = []
        self.cache = []
        
        # Process each timestep
        for t in range(seq_len):
            x_t = input_data[:, t, :]  # (batch_size, input_size)
            h_prev = h.copy()
            
            # Update gate
            z_t = self._sigmoid(
                np.dot(x_t, self.params['W_z']) + 
                np.dot(h_prev, self.params['U_z']) + 
                self.params['b_z']
            )
            
            # Reset gate
            r_t = self._sigmoid(
                np.dot(x_t, self.params['W_r']) + 
                np.dot(h_prev, self.params['U_r']) + 
                self.params['b_r']
            )
            
            # Candidate activation
            h_tilde = self._tanh(
                np.dot(x_t, self.params['W_h']) + 
                np.dot(r_t * h_prev, self.params['U_h']) + 
                self.params['b_h']
            )
            
            # Update hidden state
            h = (1 - z_t) * h_prev + z_t * h_tilde
            
            # Store for backward pass
            self.cache.append({
                'x_t': x_t,
                'h_prev': h_prev,
                'z_t': z_t,
                'r_t': r_t,
                'h_tilde': h_tilde,
                'h': h.copy()
            })
            
            outputs.append(h.copy())
        
        # Stack outputs
        output = np.stack(outputs, axis=1)  # (batch_size, seq_len, hidden_size)
        
        if not self.return_sequences:
            output = output[:, -1, :]  # (batch_size, hidden_size)
        
        if self.return_state:
            return output, h
        
        return output
    
    def backward(self, output_grad):
        """
        Args:
            output_grad: Gradient from next layer
                        (batch_size, seq_len, hidden_size) or (batch_size, hidden_size)
        
        Returns:
            input_grad: Gradient w.r.t. input (batch_size, seq_len, input_size)
        """
        batch_size, seq_len, _ = self.input.shape
        
        # If return_sequences=False, we only have gradient for last timestep
        if not self.return_sequences:
            # Expand to all timesteps with zeros except last
            grad_expanded = np.zeros((batch_size, seq_len, self.hidden_size))
            grad_expanded[:, -1, :] = output_grad
            output_grad = grad_expanded
        
        # Initialize gradients
        dh_next = np.zeros((batch_size, self.hidden_size))
        
        input_grad = np.zeros_like(self.input)
        
        # Reset parameter gradients
        for key in self.grads:
            self.grads[key] = np.zeros_like(self.params[key])
        
        # Backward through time
        for t in reversed(range(seq_len)):
            cache_t = self.cache[t]
            grad_t = output_grad[:, t, :]  # (batch_size, hidden_size)
            
            # Combine gradients
            dh = grad_t + dh_next
            
            # Gradients w.r.t. gates and candidate
            dz = dh * (cache_t['h_tilde'] - cache_t['h_prev'])
            dh_tilde = dh * cache_t['z_t']
            dh_prev_part1 = dh * (1 - cache_t['z_t'])
            dh_prev_part2 = dh_tilde * cache_t['r_t']
            
            # Gradients through activations
            dz_raw = dz * cache_t['z_t'] * (1 - cache_t['z_t'])
            
            # Gradient w.r.t. reset gate
            # dh_tilde depends on r_t through: h_tilde = tanh(W_h * x_t + U_h * (r_t * h_prev) + b_h)
            # So dL/dr = dL/dh_tilde * dh_tilde/dr = dL/dh_tilde * (1 - h_tilde^2) * U_h * h_prev
            dh_tilde_raw = dh_tilde * (1 - cache_t['h_tilde'] ** 2)
            dr = np.sum(dh_tilde_raw * np.dot(cache_t['h_prev'], self.params['U_h']), axis=1, keepdims=True)
            dr_raw = dr * cache_t['r_t'] * (1 - cache_t['r_t'])
            
            # Gradients w.r.t. input
            x_t = cache_t['x_t']
            h_prev = cache_t['h_prev']
            
            dx_t = (np.dot(dz_raw, self.params['W_z'].T) +
                   np.dot(dr_raw, self.params['W_r'].T) +
                   np.dot(dh_tilde_raw, self.params['W_h'].T))
            
            input_grad[:, t, :] = dx_t
            
            # Gradients w.r.t. parameters
            self.grads['W_z'] += np.dot(x_t.T, dz_raw) / batch_size
            self.grads['U_z'] += np.dot(h_prev.T, dz_raw) / batch_size
            self.grads['b_z'] += np.sum(dz_raw, axis=0, keepdims=True) / batch_size
            
            self.grads['W_r'] += np.dot(x_t.T, dr_raw) / batch_size
            self.grads['U_r'] += np.dot(h_prev.T, dr_raw) / batch_size
            self.grads['b_r'] += np.sum(dr_raw, axis=0, keepdims=True) / batch_size
            
            self.grads['W_h'] += np.dot(x_t.T, dh_tilde_raw) / batch_size
            self.grads['U_h'] += np.dot((cache_t['r_t'] * h_prev).T, dh_tilde_raw) / batch_size
            self.grads['b_h'] += np.sum(dh_tilde_raw, axis=0, keepdims=True) / batch_size
            
            # Update gradients for previous timestep
            dh_next = (dh_prev_part1 + 
                      np.dot(dz_raw, self.params['U_z'].T) +
                      np.dot(dr_raw, self.params['U_r'].T) +
                      np.dot(dh_tilde_raw, self.params['U_h'].T) * cache_t['r_t'])
        
        return input_grad

