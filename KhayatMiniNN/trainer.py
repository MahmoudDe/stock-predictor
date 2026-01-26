import numpy as np
import time
from .gpu_utils import to_device, to_numpy, get_xp


class Trainer:
    
    def __init__(self, network, optimizer, loss_fn):
        self.network = network
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.network.set_loss(loss_fn)
        
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        
        self.layer_optimizers = {}
        self.device_manager = network.device_manager
        self.xp = get_xp(self.device_manager)
        
        # Set device_manager for optimizer if it supports it
        if hasattr(optimizer, 'device_manager'):
            optimizer.device_manager = self.device_manager
            if hasattr(optimizer, 'xp'):
                optimizer.xp = self.xp
    
    def train_step(self, X_batch, y_batch):
        # Move data to device
        X_batch = to_device(X_batch, self.device_manager)
        y_batch = to_device(y_batch, self.device_manager)
        
        predictions = self.network.forward(X_batch)
        loss = self.network.compute_loss(predictions, y_batch)
        accuracy = self.network.compute_accuracy(predictions, y_batch)
        
        loss_grad = self.loss_fn.backward()
        self.network.backward(loss_grad)
        
        grads = self.network.get_grads()
        params = self.network.get_params()
        
        self._update_params(params, grads)
        
        # Synchronize GPU operations
        self.device_manager.synchronize()
        
        return float(loss), float(accuracy)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, 
            epochs=10, batch_size=32, verbose=True, val_interval=1):
        # Move data to device
        X_train = to_device(X_train, self.device_manager)
        y_train = to_device(y_train, self.device_manager)
        if X_val is not None:
            X_val = to_device(X_val, self.device_manager)
        if y_val is not None:
            y_val = to_device(y_val, self.device_manager)
        
        num_samples = X_train.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # Shuffle on CPU then move to device
            indices = np.random.permutation(num_samples)
            X_shuffled = to_device(X_train[indices], self.device_manager)
            y_shuffled = to_device(y_train[indices], self.device_manager)
            
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            
            self.network._set_training(True)
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                batch_loss, batch_accuracy = self.train_step(X_batch, y_batch)
                
                epoch_loss += batch_loss
                epoch_accuracy += batch_accuracy
            
            avg_loss = epoch_loss / num_batches
            avg_accuracy = epoch_accuracy / num_batches
            
            self.train_losses.append(avg_loss)
            self.train_accuracies.append(avg_accuracy)
            
            if X_val is not None and y_val is not None and (epoch + 1) % val_interval == 0:
                val_loss, val_accuracy = self.evaluate(X_val, y_val)
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_accuracy)
                
                if verbose:
                    elapsed = time.time() - start_time
                    print(f"Epoch {epoch+1}/{epochs} | "
                          f"Loss: {avg_loss:.4f} | Acc: {avg_accuracy:.2f}% | "
                          f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}% | "
                          f"Time: {elapsed:.2f}s")
            else:
                if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                    elapsed = time.time() - start_time
                    print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | "
                          f"Accuracy: {avg_accuracy:.2f}% | Time: {elapsed:.2f}s")
        
        return {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }
    
    def evaluate(self, X_test, y_test):
        # Move data to device
        X_test = to_device(X_test, self.device_manager)
        y_test = to_device(y_test, self.device_manager)
        
        self.network._set_training(False)
        predictions = self.network.forward(X_test)
        loss = self.network.compute_loss(predictions, y_test)
        accuracy = self.network.compute_accuracy(predictions, y_test)
        self.network._set_training(True)
        
        # Synchronize GPU operations
        self.device_manager.synchronize()
        
        return float(loss), float(accuracy)
    
    def predict(self, X):
        return self.network.predict(X)
    
    def _update_params(self, params, grads):
        for layer_name in params:
            if layer_name in grads:
                layer_params = params[layer_name]
                layer_grads = grads[layer_name]
                
                if layer_name not in self.layer_optimizers:
                    optimizer_type = type(self.optimizer)
                    if optimizer_type.__name__ == 'Adam':
                        self.layer_optimizers[layer_name] = optimizer_type(
                            learning_rate=self.optimizer.learning_rate,
                            device_manager=self.device_manager
                        )
                    elif optimizer_type.__name__ == 'Momentum':
                        self.layer_optimizers[layer_name] = optimizer_type(
                            learning_rate=self.optimizer.learning_rate,
                            momentum=self.optimizer.momentum,
                            device_manager=self.device_manager
                        )
                    elif optimizer_type.__name__ == 'Adagrad':
                        self.layer_optimizers[layer_name] = optimizer_type(
                            learning_rate=self.optimizer.learning_rate,
                            device_manager=self.device_manager
                        )
                    else:
                        self.layer_optimizers[layer_name] = optimizer_type(
                            learning_rate=self.optimizer.learning_rate,
                            device_manager=self.device_manager
                        )
                
                updated_params = self.layer_optimizers[layer_name].update(layer_params, layer_grads)
                
                layer = self.network.layers[layer_name]
                if hasattr(layer, 'set_params'):
                    layer.set_params(updated_params)
                else:
                    if hasattr(layer, 'params'):
                        for param_name in updated_params:
                            if param_name in layer.params:
                                layer.params[param_name] = updated_params[param_name]
