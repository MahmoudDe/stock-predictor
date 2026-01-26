"""
Example: Using GPU Acceleration with KhayatMiniNN

This example demonstrates how to use GPU acceleration for faster training.
"""

import numpy as np
from KhayatMiniNN.neural_network import NeuralNetwork
from KhayatMiniNN.layers import Dense, ReLU, Sigmoid
from KhayatMiniNN.optimizers import Adam
from KhayatMiniNN.losses import BinaryCrossEntropy
from KhayatMiniNN.trainer import Trainer
from KhayatMiniNN.gpu_utils import set_device

# ============================================================
# OPTION 1: Automatic GPU Detection (Recommended)
# ============================================================
# The network will automatically use GPU if CuPy is installed
# and a GPU is available, otherwise it falls back to CPU.

print("=" * 60)
print("GPU Acceleration Example")
print("=" * 60)

# Create network with automatic device selection
network = NeuralNetwork(name="GPU_Network", device='auto')

# Add layers (they will automatically use GPU if available)
network.add_layer(Dense(100, 64), name="dense1")
network.add_layer(ReLU(), name="relu1")
network.add_layer(Dense(64, 32), name="dense2")
network.add_layer(ReLU(), name="relu2")
network.add_layer(Dense(32, 1), name="dense3")
network.add_layer(Sigmoid(), name="sigmoid")

# Create optimizer and loss (will use GPU automatically)
optimizer = Adam(learning_rate=0.001)
loss_fn = BinaryCrossEntropy(from_logits=False)

# Create trainer
trainer = Trainer(network, optimizer, loss_fn)

# Generate sample data
print("\nGenerating sample data...")
X_train = np.random.randn(1000, 100).astype(np.float32)
y_train = np.random.randint(0, 2, (1000, 1)).astype(np.float32)

X_val = np.random.randn(200, 100).astype(np.float32)
y_val = np.random.randint(0, 2, (200, 1)).astype(np.float32)

print(f"Training data shape: {X_train.shape}")
print(f"Device: {network.device_manager.device_type}")

# Train the network
print("\nStarting training...")
history = trainer.fit(
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    epochs=10,
    batch_size=32,
    verbose=True
)

print("\nTraining completed!")
print(f"Final validation accuracy: {history['val_accuracies'][-1]:.2f}%")

# ============================================================
# OPTION 2: Force GPU Usage
# ============================================================
# Uncomment to force GPU usage (will raise error if GPU not available)
# network_gpu = NeuralNetwork(name="GPU_Network", device='gpu')

# ============================================================
# OPTION 3: Force CPU Usage
# ============================================================
# Uncomment to force CPU usage even if GPU is available
# network_cpu = NeuralNetwork(name="CPU_Network", device='cpu')

# ============================================================
# NOTES:
# ============================================================
# 1. Install CuPy for GPU support:
#    - For CUDA 11.x: pip install cupy-cuda11x
#    - For CUDA 12.x: pip install cupy-cuda12x
#    - Check your CUDA version: nvidia-smi
#
# 2. GPU acceleration provides significant speedup for:
#    - Large batch sizes
#    - Large networks
#    - Complex operations (LSTM, Conv1D, etc.)
#
# 3. For small networks or small batches, CPU might be faster
#    due to GPU overhead
#
# 4. Data is automatically transferred to/from GPU as needed
#
# 5. All NumPy operations are replaced with GPU-equivalent operations
#    when GPU is enabled
