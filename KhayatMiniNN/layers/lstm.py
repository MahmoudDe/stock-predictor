import numpy as np
from .base import Layer
from ..utils import initialize_weights
from ..gpu_utils import get_xp, to_device


class LSTM(Layer):
    """
    Long Short-Term Memory (LSTM) layer for sequential data.
    
    Processes sequences of shape (batch_size, sequence_length, input_size)
    and outputs (batch_size, sequence_length, hidden_size) or 
    (batch_size, hidden_size) depending on return_sequences.
    """
    
    def __init__(self, input_size, hidden_size, return_sequences=False, 
                 return_state=False, name="LSTM", device_manager=None):
        """
        Args:
            input_size: Size of input features at each timestep
            hidden_size: Size of hidden state
            return_sequences: If True, return all timesteps. If False, return only last.
            return_state: If True, also return hidden and cell states
            device_manager: DeviceManager instance for GPU support
        """
        super().__init__(name)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.device_manager = device_manager
        self.xp = get_xp(device_manager) if device_manager else np
        
        # Initialize weights for forget, input, output gates and cell candidate
        # W_f, W_i, W_o, W_c: (input_size, hidden_size)
        # U_f, U_i, U_o, U_c: (hidden_size, hidden_size)
        # b_f, b_i, b_o, b_c: (hidden_size,)
        
        # Forget gate weights
        w_f = initialize_weights((input_size, hidden_size), method='xavier')
        u_f = initialize_weights((hidden_size, hidden_size), method='xavier')
        self.params['W_f'] = to_device(w_f, device_manager) if device_manager else w_f
        self.params['U_f'] = to_device(u_f, device_manager) if device_manager else u_f
        self.params['b_f'] = self.xp.zeros((1, hidden_size))
        
        # Input gate weights
        w_i = initialize_weights((input_size, hidden_size), method='xavier')
        u_i = initialize_weights((hidden_size, hidden_size), method='xavier')
        self.params['W_i'] = to_device(w_i, device_manager) if device_manager else w_i
        self.params['U_i'] = to_device(u_i, device_manager) if device_manager else u_i
        self.params['b_i'] = self.xp.zeros((1, hidden_size))
        
        # Output gate weights
        w_o = initialize_weights((input_size, hidden_size), method='xavier')
        u_o = initialize_weights((hidden_size, hidden_size), method='xavier')
        self.params['W_o'] = to_device(w_o, device_manager) if device_manager else w_o
        self.params['U_o'] = to_device(u_o, device_manager) if device_manager else u_o
        self.params['b_o'] = self.xp.zeros((1, hidden_size))
        
        # Cell candidate weights
        w_c = initialize_weights((input_size, hidden_size), method='xavier')
        u_c = initialize_weights((hidden_size, hidden_size), method='xavier')
        self.params['W_c'] = to_device(w_c, device_manager) if device_manager else w_c
        self.params['U_c'] = to_device(u_c, device_manager) if device_manager else u_c
        self.params['b_c'] = self.xp.zeros((1, hidden_size))
        
        # Initialize gradients
        for key in ['W_f', 'U_f', 'b_f', 'W_i', 'U_i', 'b_i', 
                   'W_o', 'U_o', 'b_o', 'W_c', 'U_c', 'b_c']:
            self.grads[key] = self.xp.zeros_like(self.params[key])
        
        # Cache for backward pass
        self.cache = []
    
    def _sigmoid(self, x):
        """Sigmoid activation."""
        x_clipped = self.xp.clip(x, -500, 500)
        return 1 / (1 + self.xp.exp(-x_clipped))
    
    def _tanh(self, x):
        """Tanh activation."""
        return self.xp.tanh(x)
    
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
        
        # Initialize hidden and cell states
        h = self.xp.zeros((batch_size, self.hidden_size))
        c = self.xp.zeros((batch_size, self.hidden_size))
        
        outputs = []
        self.cache = []
        
        # Process each timestep
        for t in range(seq_len):
            x_t = input_data[:, t, :]  # (batch_size, input_size)
            
            # Forget gate
            f_t = self._sigmoid(
                self.xp.dot(x_t, self.params['W_f']) + 
                self.xp.dot(h, self.params['U_f']) + 
                self.params['b_f']
            )
            
            # Input gate
            i_t = self._sigmoid(
                self.xp.dot(x_t, self.params['W_i']) + 
                self.xp.dot(h, self.params['U_i']) + 
                self.params['b_i']
            )
            
            # Output gate
            o_t = self._sigmoid(
                self.xp.dot(x_t, self.params['W_o']) + 
                self.xp.dot(h, self.params['U_o']) + 
                self.params['b_o']
            )
            
            # Cell candidate
            c_tilde = self._tanh(
                self.xp.dot(x_t, self.params['W_c']) + 
                self.xp.dot(h, self.params['U_c']) + 
                self.params['b_c']
            )
            
            # Update cell state
            c = f_t * c + i_t * c_tilde
            
            # Update hidden state
            h = o_t * self._tanh(c)
            
            # Store for backward pass
            h_prev = self.xp.zeros((batch_size, self.hidden_size)) if t == 0 else self.cache[-1]['h']
            c_prev = self.xp.zeros((batch_size, self.hidden_size)) if t == 0 else self.cache[-1]['c']
            
            self.cache.append({
                'x_t': x_t,
                'h_prev': h_prev,
                'c_prev': c_prev,
                'f_t': f_t,
                'i_t': i_t,
                'o_t': o_t,
                'c_tilde': c_tilde,
                'c': c.copy(),
                'h': h.copy()
            })
            
            outputs.append(h.copy())
        
        # Stack outputs
        output = self.xp.stack(outputs, axis=1)  # (batch_size, seq_len, hidden_size)
        
        if not self.return_sequences:
            output = output[:, -1, :]  # (batch_size, hidden_size)
        
        if self.return_state:
            return output, h, c
        
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
            grad_expanded = self.xp.zeros((batch_size, seq_len, self.hidden_size))
            grad_expanded[:, -1, :] = output_grad
            output_grad = grad_expanded
        
        # Initialize gradients
        dh_next = self.xp.zeros((batch_size, self.hidden_size))
        dc_next = self.xp.zeros((batch_size, self.hidden_size))
        
        input_grad = self.xp.zeros_like(self.input)
        
        # Reset parameter gradients
        for key in self.grads:
            self.grads[key] = self.xp.zeros_like(self.params[key])
        
        # Backward through time
        for t in reversed(range(seq_len)):
            cache_t = self.cache[t]
            grad_t = output_grad[:, t, :]  # (batch_size, hidden_size)
            
            # Combine gradients
            dh = grad_t + dh_next
            dc = dc_next + dh * cache_t['o_t'] * (1 - self.xp.tanh(cache_t['c']) ** 2)
            
            # Gradients w.r.t. gates
            do = dh * self.xp.tanh(cache_t['c'])
            di = dc * cache_t['c_tilde']
            df = dc * cache_t['c_prev']
            dc_tilde = dc * cache_t['i_t']
            
            # Gradients through activations
            do_raw = do * cache_t['o_t'] * (1 - cache_t['o_t'])
            di_raw = di * cache_t['i_t'] * (1 - cache_t['i_t'])
            df_raw = df * cache_t['f_t'] * (1 - cache_t['f_t'])
            dc_tilde_raw = dc_tilde * (1 - cache_t['c_tilde'] ** 2)
            
            # Gradients w.r.t. input
            x_t = cache_t['x_t']
            h_prev = cache_t['h_prev'] if t > 0 else self.xp.zeros((batch_size, self.hidden_size))
            
            dx_t = (self.xp.dot(do_raw, self.params['W_o'].T) +
                   self.xp.dot(di_raw, self.params['W_i'].T) +
                   self.xp.dot(df_raw, self.params['W_f'].T) +
                   self.xp.dot(dc_tilde_raw, self.params['W_c'].T))
            
            input_grad[:, t, :] = dx_t
            
            # Gradients w.r.t. parameters
            self.grads['W_o'] += self.xp.dot(x_t.T, do_raw) / batch_size
            self.grads['U_o'] += self.xp.dot(h_prev.T, do_raw) / batch_size
            self.grads['b_o'] += self.xp.sum(do_raw, axis=0, keepdims=True) / batch_size
            
            self.grads['W_i'] += self.xp.dot(x_t.T, di_raw) / batch_size
            self.grads['U_i'] += self.xp.dot(h_prev.T, di_raw) / batch_size
            self.grads['b_i'] += self.xp.sum(di_raw, axis=0, keepdims=True) / batch_size
            
            self.grads['W_f'] += self.xp.dot(x_t.T, df_raw) / batch_size
            self.grads['U_f'] += self.xp.dot(h_prev.T, df_raw) / batch_size
            self.grads['b_f'] += self.xp.sum(df_raw, axis=0, keepdims=True) / batch_size
            
            self.grads['W_c'] += self.xp.dot(x_t.T, dc_tilde_raw) / batch_size
            self.grads['U_c'] += self.xp.dot(h_prev.T, dc_tilde_raw) / batch_size
            self.grads['b_c'] += self.xp.sum(dc_tilde_raw, axis=0, keepdims=True) / batch_size
            
            # Update gradients for previous timestep
            dh_next = (self.xp.dot(do_raw, self.params['U_o'].T) +
                      self.xp.dot(di_raw, self.params['U_i'].T) +
                      self.xp.dot(df_raw, self.params['U_f'].T) +
                      self.xp.dot(dc_tilde_raw, self.params['U_c'].T))
            
            dc_next = dc * cache_t['f_t']
        
        return input_grad

