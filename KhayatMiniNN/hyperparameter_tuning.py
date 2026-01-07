import numpy as np
from itertools import product
from .neural_network import NeuralNetwork
from .trainer import Trainer
from .layers import Dense, ReLU, Sigmoid, Tanh, Linear
from .losses import MeanSquaredError, SoftmaxCrossEntropy
from .optimizers import SGD, Adam, Momentum, Adagrad
from .regularization import Dropout, BatchNormalization
from .utils import one_hot_encode


class HyperparameterTuning:
    
    def __init__(self, X_train, y_train, X_val=None, y_val=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val if X_val is not None else X_train
        self.y_val = y_val if y_val is not None else y_train
        
        self.results = []
        self.best_params = None
        self.best_score = -np.inf
    
    def _create_optimizer(self, optimizer_type, learning_rate):
        if optimizer_type == 'SGD':
            return SGD(learning_rate=learning_rate)
        elif optimizer_type == 'Adam':
            return Adam(learning_rate=learning_rate)
        elif optimizer_type == 'Momentum':
            return Momentum(learning_rate=learning_rate, momentum=0.9)
        elif optimizer_type == 'Adagrad':
            return Adagrad(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    def _create_activation(self, activation_name):
        if activation_name == 'ReLU':
            return ReLU()
        elif activation_name == 'Sigmoid':
            return Sigmoid()
        elif activation_name == 'Tanh':
            return Tanh()
        elif activation_name == 'Linear':
            return Linear()
        else:
            raise ValueError(f"Unknown activation: {activation_name}")
    
    def _build_network(self, input_size, output_size, layer_sizes, 
                      activations, dropout_rate=None, use_batch_norm=False):
        network = NeuralNetwork(name="TunedNetwork")
        current_size = input_size
        
        for i, (layer_size, activation_name) in enumerate(zip(layer_sizes, activations)):
            network.add_layer(Dense(current_size, layer_size), name=f"dense_{i}")
            
            if use_batch_norm:
                network.add_layer(BatchNormalization(layer_size), name=f"batchnorm_{i}")
            
            activation = self._create_activation(activation_name)
            network.add_layer(activation, name=f"activation_{i}")
            
            if dropout_rate is not None and dropout_rate > 0:
                network.add_layer(Dropout(dropout_rate=dropout_rate), name=f"dropout_{i}")
            
            current_size = layer_size
        
        network.add_layer(Dense(current_size, output_size), name="output")
        return network
    
    def _determine_loss_function(self, y_train):
        if y_train.ndim == 1 or (y_train.ndim == 2 and y_train.shape[1] == 1):
            num_classes = len(np.unique(y_train))
            if num_classes == 2:
                return MeanSquaredError()
            else:
                return SoftmaxCrossEntropy()
        elif y_train.ndim == 2 and y_train.shape[1] > 1:
            if np.all((y_train == 0) | (y_train == 1)) and np.allclose(y_train.sum(axis=1), 1):
                return SoftmaxCrossEntropy()
            else:
                return MeanSquaredError()
        else:
            return MeanSquaredError()
    
    def grid_search(self, param_grid, verbose=True):
        # Determine input and output sizes
        input_size = self.X_train.shape[1]
        
        # Determine output size and loss function
        if self.y_train.ndim == 1:
            output_size = len(np.unique(self.y_train))
            # Convert to one-hot if needed
            y_train_encoded = one_hot_encode(self.y_train, output_size)
            y_val_encoded = one_hot_encode(self.y_val, output_size) if self.y_val.ndim == 1 else self.y_val
        else:
            output_size = self.y_train.shape[1]
            y_train_encoded = self.y_train
            y_val_encoded = self.y_val
        
        # Get loss function
        loss_fn = self._determine_loss_function(self.y_train)
        
        # Extract parameter lists
        learning_rates = param_grid.get('learning_rates', [0.01])
        batch_sizes = param_grid.get('batch_sizes', [32])
        epochs_list = param_grid.get('epochs', [10])
        optimizers = param_grid.get('optimizers', ['SGD'])
        layer_sizes_list = param_grid.get('layer_sizes', [[64]])
        activations_list = param_grid.get('activations', [['ReLU']])
        dropout_rates = param_grid.get('dropout_rates', [None])
        use_batch_norm_list = param_grid.get('use_batch_norm', [False])
        
        # Generate all combinations
        total_combinations = (len(learning_rates) * len(batch_sizes) * len(epochs_list) *
                             len(optimizers) * len(layer_sizes_list) * len(activations_list) *
                             len(dropout_rates) * len(use_batch_norm_list))
        
        if verbose:
            print(f"Testing {total_combinations} combinations...")
            print(f"Input: {input_size}, Output: {output_size}\n")
        
        combination_num = 0
        
        # Try all combinations
        for lr, batch_size, epochs, opt_type, layer_sizes, activations, dropout_rate, use_bn in product(
            learning_rates, batch_sizes, epochs_list, optimizers, 
            layer_sizes_list, activations_list, dropout_rates, use_batch_norm_list
        ):
            combination_num += 1
            
            if len(activations) != len(layer_sizes):
                if verbose:
                    print(f"Skip {combination_num}/{total_combinations}: activations mismatch")
                continue
            
            try:
                # Build network
                network = self._build_network(
                    input_size=input_size,
                    output_size=output_size,
                    layer_sizes=layer_sizes,
                    activations=activations,
                    dropout_rate=dropout_rate,
                    use_batch_norm=use_bn
                )
                
                # Create optimizer
                optimizer = self._create_optimizer(opt_type, lr)
                
                # Create trainer
                trainer = Trainer(network, optimizer, loss_fn)
                
                # Train
                history = trainer.fit(
                    X_train=self.X_train,
                    y_train=y_train_encoded,
                    X_val=self.X_val,
                    y_val=y_val_encoded,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=False
                )
                
                # Get final validation accuracy
                val_accuracy = history['val_accuracies'][-1] if history['val_accuracies'] else 0
                train_accuracy = history['train_accuracies'][-1] if history['train_accuracies'] else 0
                val_loss = history['val_losses'][-1] if history['val_losses'] else float('inf')
                
                # Store results
                result = {
                    'learning_rate': lr,
                    'batch_size': batch_size,
                    'epochs': epochs,
                    'optimizer': opt_type,
                    'layer_sizes': layer_sizes,
                    'activations': activations,
                    'dropout_rate': dropout_rate,
                    'use_batch_norm': use_bn,
                    'val_accuracy': val_accuracy,
                    'train_accuracy': train_accuracy,
                    'val_loss': val_loss
                }
                self.results.append(result)
                
                # Update best if better
                if val_accuracy > self.best_score:
                    self.best_score = val_accuracy
                    self.best_params = result.copy()
                
                if verbose:
                    print(f"{combination_num}/{total_combinations}: lr={lr:.4f}, batch={batch_size}, "
                          f"epochs={epochs}, opt={opt_type}, val_acc={val_accuracy:.2f}%")
            
            except Exception as e:
                if verbose:
                    print(f"{combination_num}/{total_combinations}: Error - {str(e)}")
                continue
        
        if verbose:
            print(f"\nDone! Best val acc: {self.best_score:.2f}%")
            print("Best params:")
            for key, value in self.best_params.items():
                if key not in ['val_accuracy', 'train_accuracy', 'val_loss']:
                    print(f"  {key}: {value}")
            print()
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'all_results': self.results
        }
    
    def get_best_model(self, X_train, y_train):
        if self.best_params is None:
            raise ValueError("No best parameters found. Run grid_search() first.")
        
        # Determine sizes
        input_size = X_train.shape[1]
        if y_train.ndim == 1:
            output_size = len(np.unique(y_train))
        else:
            output_size = y_train.shape[1]
        
        # Build network with best parameters
        network = self._build_network(
            input_size=input_size,
            output_size=output_size,
            layer_sizes=self.best_params['layer_sizes'],
            activations=self.best_params['activations'],
            dropout_rate=self.best_params['dropout_rate'],
            use_batch_norm=self.best_params['use_batch_norm']
        )
        
        # Create optimizer
        optimizer = self._create_optimizer(
            self.best_params['optimizer'],
            self.best_params['learning_rate']
        )
        
        # Create loss function
        loss_fn = self._determine_loss_function(y_train)
        
        return network, optimizer, loss_fn
    
    def print_results_summary(self, top_n=5):
        if not self.results:
            print("No results yet")
            return
        
        sorted_results = sorted(self.results, key=lambda x: x['val_accuracy'], reverse=True)
        
        print(f"\nTop {top_n} results:")
        for i, result in enumerate(sorted_results[:top_n], 1):
            print(f"{i}. Val acc: {result['val_accuracy']:.2f}%, "
                  f"lr={result['learning_rate']:.4f}, "
                  f"batch={result['batch_size']}, "
                  f"epochs={result['epochs']}, "
                  f"opt={result['optimizer']}")
        print()

