import numpy as np
from collections import OrderedDict
from .gpu_utils import get_device_manager, get_xp, to_device, to_numpy


class NeuralNetwork:
    
    def __init__(self, name="NeuralNetwork", device='auto'):
        self.name = name
        self.layers = OrderedDict()
        self.loss_fn = None
        self.training = True
        self.device_manager = get_device_manager(device=device)
        self.xp = get_xp(self.device_manager)
    
    def add_layer(self, layer, name=None):
        if name is None:
            name = f"{layer.name}_{len(self.layers)}"
        # Set device_manager for layer if it supports it
        if hasattr(layer, 'device_manager'):
            layer.device_manager = self.device_manager
            if hasattr(layer, 'xp'):
                layer.xp = self.xp
        self.layers[name] = layer
    
    def set_loss(self, loss_fn):
        self.loss_fn = loss_fn
        # Set device_manager for loss function if it supports it
        if hasattr(loss_fn, 'device_manager'):
            loss_fn.device_manager = self.device_manager
            if hasattr(loss_fn, 'xp'):
                loss_fn.xp = self.xp
    
    def forward(self, input_data):
        # Ensure input is on correct device
        input_data = to_device(input_data, self.device_manager)
        output = input_data
        for layer in self.layers.values():
            output = layer.forward(output)
        return output
    
    def backward(self, output_grad):
        grad = output_grad
        for layer in reversed(self.layers.values()):
            grad = layer.backward(grad)
        return grad
    
    def predict(self, input_data):
        self._set_training(False)
        output = self.forward(input_data)
        self._set_training(True)
        return output
    
    def compute_loss(self, predictions, targets):
        if self.loss_fn is None:
            raise ValueError("Loss function not set. Use set_loss() first.")
        loss = self.loss_fn.forward(predictions, targets)
        return loss
    
    def compute_accuracy(self, predictions, targets):
        xp = self.xp
        # Convert to numpy for accuracy computation (argmax and comparison)
        pred_np = to_numpy(predictions, self.device_manager)
        targ_np = to_numpy(targets, self.device_manager)
        
        if targ_np.ndim == 2 and targ_np.shape[1] > 1:
            # Multi-class classification
            target_classes = np.argmax(targ_np, axis=1)
            pred_classes = np.argmax(pred_np, axis=1)
        else:
            # Binary classification - flatten arrays if they're (N, 1)
            if targ_np.ndim == 2 and targ_np.shape[1] == 1:
                # Binary classification with shape (N, 1) - flatten to (N,)
                target_classes = targ_np.flatten().astype(int)
            else:
                # Already 1D
                target_classes = targ_np.astype(int)
            
            if pred_np.ndim == 2 and pred_np.shape[1] == 1:
                # Binary classification with shape (N, 1) - use threshold
                pred_classes = (pred_np.flatten() > 0.5).astype(int)
            elif pred_np.ndim == 2:
                # Multi-class with shape (N, num_classes)
                pred_classes = np.argmax(pred_np, axis=1)
            else:
                # Already 1D
                pred_classes = (pred_np > 0.5).astype(int)
        
        accuracy = np.mean(pred_classes == target_classes) * 100
        return accuracy
    
    def get_params(self):
        all_params = {}
        for layer_name, layer in self.layers.items():
            if hasattr(layer, 'get_params'):
                params = layer.get_params()
                if params:
                    all_params[layer_name] = params
        return all_params
    
    def get_grads(self):
        all_grads = {}
        for layer_name, layer in self.layers.items():
            if hasattr(layer, 'get_grads'):
                grads = layer.get_grads()
                if grads:
                    all_grads[layer_name] = grads
        return all_grads
    
    def set_params(self, params_dict):
        for layer_name, params in params_dict.items():
            if layer_name in self.layers:
                if hasattr(self.layers[layer_name], 'set_params'):
                    self.layers[layer_name].set_params(params)
    
    def _set_training(self, training=True):
        for layer in self.layers.values():
            if hasattr(layer, 'set_training'):
                layer.set_training(training)
        self.training = training
    
    def get_layer_params(self, layer_name):
        if layer_name in self.layers:
            layer = self.layers[layer_name]
            if hasattr(layer, 'get_params'):
                return layer.get_params()
        return None
    
    def summary(self):
        print(f"\n{'='*60}")
        print(f"Network: {self.name}")
        print(f"{'='*60}")
        print(f"{'Layer Name':<20} {'Type':<20} {'Parameters':<20}")
        print(f"{'-'*60}")
        
        total_params = 0
        for layer_name, layer in self.layers.items():
            layer_type = layer.__class__.__name__
            params = layer.get_params() if hasattr(layer, 'get_params') else {}
            
            if params:
                param_count = sum(np.prod(p.shape) for p in params.values())
                total_params += param_count
                print(f"{layer_name:<20} {layer_type:<20} {param_count:<20}")
            else:
                print(f"{layer_name:<20} {layer_type:<20} {'0':<20}")
        
        print(f"{'-'*60}")
        print(f"Total parameters: {total_params}")
        print(f"{'='*60}\n")
