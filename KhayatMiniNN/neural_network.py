import numpy as np
from collections import OrderedDict


class NeuralNetwork:
    
    def __init__(self, name="NeuralNetwork"):
        self.name = name
        self.layers = OrderedDict()
        self.loss_fn = None
        self.training = True
    
    def add_layer(self, layer, name=None):
        if name is None:
            name = f"{layer.name}_{len(self.layers)}"
        self.layers[name] = layer
    
    def set_loss(self, loss_fn):
        self.loss_fn = loss_fn
    
    def forward(self, input_data):
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
        if targets.ndim == 2 and targets.shape[1] > 1:
            target_classes = np.argmax(targets, axis=1)
            pred_classes = np.argmax(predictions, axis=1)
        else:
            target_classes = np.argmax(targets, axis=1) if targets.ndim == 2 else targets
            pred_classes = np.argmax(predictions, axis=1) if predictions.ndim == 2 else (predictions > 0.5).astype(int).flatten()
        
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
