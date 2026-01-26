"""
Test script for new time series components in KhayatMiniNN.

Tests:
- BinaryCrossEntropy loss
- LSTM layer
- GRU layer
- Conv1D layer
- Pooling layers
- Embedding layer
- Sequence utilities
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from KhayatMiniNN.layers import (
    LSTM, GRU, Conv1D, MaxPooling1D, AveragePooling1D, Embedding,
    Dense, ReLU, Sigmoid
)
from KhayatMiniNN.losses import BinaryCrossEntropy, MeanSquaredError
from KhayatMiniNN.optimizers import Adam, SGD
from KhayatMiniNN.neural_network import NeuralNetwork
from KhayatMiniNN.trainer import Trainer
from KhayatMiniNN.utils.sequence_utils import (
    create_sequences, pad_sequences, split_time_series,
    create_lagged_features, create_rolling_features
)


def test_binary_crossentropy():
    """Test BinaryCrossEntropy loss function."""
    print("=" * 60)
    print("Testing BinaryCrossEntropy Loss")
    print("=" * 60)
    
    # Test with logits
    loss_fn = BinaryCrossEntropy(from_logits=True)
    
    batch_size = 10
    predictions = np.random.randn(batch_size, 1)
    targets = np.random.randint(0, 2, (batch_size, 1)).astype(float)
    
    loss = loss_fn.forward(predictions, targets)
    grad = loss_fn.backward()
    
    print(f"✓ Loss (from_logits=True): {loss:.4f}")
    print(f"✓ Gradient shape: {grad.shape}")
    assert grad.shape == predictions.shape, "Gradient shape mismatch!"
    
    # Test with sigmoid outputs
    loss_fn2 = BinaryCrossEntropy(from_logits=False)
    sigmoid_preds = 1 / (1 + np.exp(-predictions))
    loss2 = loss_fn2.forward(sigmoid_preds, targets)
    grad2 = loss_fn2.backward()
    
    print(f"✓ Loss (from_logits=False): {loss2:.4f}")
    print(f"✓ Gradient shape: {grad2.shape}")
    assert grad2.shape == sigmoid_preds.shape, "Gradient shape mismatch!"
    
    print("✓ BinaryCrossEntropy test passed!\n")


def test_lstm():
    """Test LSTM layer."""
    print("=" * 60)
    print("Testing LSTM Layer")
    print("=" * 60)
    
    batch_size = 4
    seq_len = 10
    input_size = 5
    hidden_size = 8
    
    lstm = LSTM(input_size, hidden_size, return_sequences=True)
    
    # Create input: (batch_size, seq_len, input_size)
    X = np.random.randn(batch_size, seq_len, input_size)
    
    # Forward pass
    output = lstm.forward(X)
    print(f"✓ Input shape: {X.shape}")
    print(f"✓ Output shape (return_sequences=True): {output.shape}")
    assert output.shape == (batch_size, seq_len, hidden_size), "Output shape mismatch!"
    
    # Test return_sequences=False
    lstm2 = LSTM(input_size, hidden_size, return_sequences=False)
    output2 = lstm2.forward(X)
    print(f"✓ Output shape (return_sequences=False): {output2.shape}")
    assert output2.shape == (batch_size, hidden_size), "Output shape mismatch!"
    
    # Test backward pass
    output_grad = np.random.randn(batch_size, seq_len, hidden_size)
    input_grad = lstm.backward(output_grad)
    print(f"✓ Input gradient shape: {input_grad.shape}")
    assert input_grad.shape == X.shape, "Input gradient shape mismatch!"
    
    # Check gradients are computed
    assert np.any(lstm.grads['W_f'] != 0), "Gradients not computed!"
    print("✓ LSTM test passed!\n")


def test_gru():
    """Test GRU layer."""
    print("=" * 60)
    print("Testing GRU Layer")
    print("=" * 60)
    
    batch_size = 4
    seq_len = 10
    input_size = 5
    hidden_size = 8
    
    gru = GRU(input_size, hidden_size, return_sequences=True)
    
    # Create input: (batch_size, seq_len, input_size)
    X = np.random.randn(batch_size, seq_len, input_size)
    
    # Forward pass
    output = gru.forward(X)
    print(f"✓ Input shape: {X.shape}")
    print(f"✓ Output shape (return_sequences=True): {output.shape}")
    assert output.shape == (batch_size, seq_len, hidden_size), "Output shape mismatch!"
    
    # Test return_sequences=False
    gru2 = GRU(input_size, hidden_size, return_sequences=False)
    output2 = gru2.forward(X)
    print(f"✓ Output shape (return_sequences=False): {output2.shape}")
    assert output2.shape == (batch_size, hidden_size), "Output shape mismatch!"
    
    # Test backward pass
    output_grad = np.random.randn(batch_size, seq_len, hidden_size)
    input_grad = gru.backward(output_grad)
    print(f"✓ Input gradient shape: {input_grad.shape}")
    assert input_grad.shape == X.shape, "Input gradient shape mismatch!"
    
    # Check gradients are computed
    assert np.any(gru.grads['W_z'] != 0), "Gradients not computed!"
    print("✓ GRU test passed!\n")


def test_conv1d():
    """Test Conv1D layer."""
    print("=" * 60)
    print("Testing Conv1D Layer")
    print("=" * 60)
    
    batch_size = 4
    seq_len = 20
    input_channels = 5
    output_channels = 8
    kernel_size = 3
    
    conv = Conv1D(input_channels, output_channels, kernel_size, stride=1, padding='valid')
    
    # Create input: (batch_size, seq_len, input_channels)
    X = np.random.randn(batch_size, seq_len, input_channels)
    
    # Forward pass
    output = conv.forward(X)
    expected_len = seq_len - kernel_size + 1
    print(f"✓ Input shape: {X.shape}")
    print(f"✓ Output shape (valid padding): {output.shape}")
    assert output.shape == (batch_size, expected_len, output_channels), "Output shape mismatch!"
    
    # Test 'same' padding
    conv2 = Conv1D(input_channels, output_channels, kernel_size, stride=1, padding='same')
    output2 = conv2.forward(X)
    print(f"✓ Output shape (same padding): {output2.shape}")
    assert output2.shape[1] == seq_len, "Same padding should maintain sequence length!"
    
    # Test backward pass
    output_grad = np.random.randn(*output.shape)
    input_grad = conv.backward(output_grad)
    print(f"✓ Input gradient shape: {input_grad.shape}")
    assert input_grad.shape == X.shape, "Input gradient shape mismatch!"
    
    # Check gradients are computed
    assert np.any(conv.grads['W'] != 0), "Gradients not computed!"
    print("✓ Conv1D test passed!\n")


def test_pooling():
    """Test Pooling layers."""
    print("=" * 60)
    print("Testing Pooling Layers")
    print("=" * 60)
    
    batch_size = 4
    seq_len = 20
    channels = 8
    pool_size = 2
    
    # Test MaxPooling1D
    max_pool = MaxPooling1D(pool_size=pool_size, stride=2)
    X = np.random.randn(batch_size, seq_len, channels)
    
    output = max_pool.forward(X)
    expected_len = (seq_len - pool_size) // 2 + 1
    print(f"✓ MaxPooling input shape: {X.shape}")
    print(f"✓ MaxPooling output shape: {output.shape}")
    assert output.shape == (batch_size, expected_len, channels), "Output shape mismatch!"
    
    # Test backward
    output_grad = np.random.randn(*output.shape)
    input_grad = max_pool.backward(output_grad)
    print(f"✓ MaxPooling input gradient shape: {input_grad.shape}")
    assert input_grad.shape == X.shape, "Input gradient shape mismatch!"
    
    # Test AveragePooling1D
    avg_pool = AveragePooling1D(pool_size=pool_size, stride=2)
    output2 = avg_pool.forward(X)
    print(f"✓ AveragePooling output shape: {output2.shape}")
    assert output2.shape == (batch_size, expected_len, channels), "Output shape mismatch!"
    
    output_grad2 = np.random.randn(*output2.shape)
    input_grad2 = avg_pool.backward(output_grad2)
    print(f"✓ AveragePooling input gradient shape: {input_grad2.shape}")
    assert input_grad2.shape == X.shape, "Input gradient shape mismatch!"
    
    print("✓ Pooling layers test passed!\n")


def test_embedding():
    """Test Embedding layer."""
    print("=" * 60)
    print("Testing Embedding Layer")
    print("=" * 60)
    
    vocab_size = 100
    embedding_dim = 16
    batch_size = 4
    seq_len = 10
    
    embedding = Embedding(vocab_size, embedding_dim)
    
    # Create input indices: (batch_size, seq_len)
    X = np.random.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass
    output = embedding.forward(X)
    print(f"✓ Input shape: {X.shape}")
    print(f"✓ Output shape: {output.shape}")
    assert output.shape == (batch_size, seq_len, embedding_dim), "Output shape mismatch!"
    
    # Test 1D input
    X_1d = np.random.randint(0, vocab_size, (batch_size,))
    output_1d = embedding.forward(X_1d)
    print(f"✓ 1D Input shape: {X_1d.shape}")
    print(f"✓ 1D Output shape: {output_1d.shape}")
    assert output_1d.shape == (batch_size, embedding_dim), "1D output shape mismatch!"
    
    # Test backward pass
    output_grad = np.random.randn(batch_size, seq_len, embedding_dim)
    input_grad = embedding.backward(output_grad)
    print(f"✓ Embedding gradients computed: {np.any(embedding.grads['W'] != 0)}")
    
    print("✓ Embedding test passed!\n")


def test_sequence_utils():
    """Test sequence processing utilities."""
    print("=" * 60)
    print("Testing Sequence Utilities")
    print("=" * 60)
    
    # Test create_sequences
    num_samples = 100
    num_features = 5
    data = np.random.randn(num_samples, num_features)
    sequence_length = 10
    prediction_horizon = 1
    
    X, y = create_sequences(data, sequence_length, prediction_horizon)
    print(f"✓ create_sequences - X shape: {X.shape}, y shape: {y.shape}")
    assert X.shape[0] == num_samples - sequence_length - prediction_horizon + 1
    assert X.shape[1] == sequence_length
    assert X.shape[2] == num_features
    
    # Test pad_sequences
    sequences = [
        np.random.randn(5, 3),
        np.random.randn(8, 3),
        np.random.randn(3, 3)
    ]
    padded, lengths = pad_sequences(sequences, max_length=10, padding='post')
    print(f"✓ pad_sequences - padded shape: {padded.shape}, lengths: {lengths}")
    assert padded.shape == (3, 10, 3)
    assert np.all(lengths == [5, 8, 3])
    
    # Test split_time_series
    train_data, val_data, test_data = split_time_series(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    print(f"✓ split_time_series - train: {train_data.shape}, val: {val_data.shape}, test: {test_data.shape}")
    assert train_data.shape[0] == int(num_samples * 0.7)
    
    # Test create_lagged_features
    lagged = create_lagged_features(data[:50], lags=[1, 2, 3])
    print(f"✓ create_lagged_features - shape: {lagged.shape}")
    assert lagged.shape[1] == num_features * 4  # original + 3 lags
    
    # Test create_rolling_features
    rolling = create_rolling_features(data[:50], windows=[5, 10], functions=['mean', 'std'])
    print(f"✓ create_rolling_features - shape: {rolling.shape}")
    assert rolling.shape[1] == num_features * 2 * 2  # 2 windows * 2 functions
    
    print("✓ Sequence utilities test passed!\n")


def test_integration():
    """Test integration of new components with existing library."""
    print("=" * 60)
    print("Testing Integration with Existing Library")
    print("=" * 60)
    
    # Create a simple LSTM-based network for binary classification
    batch_size = 8
    seq_len = 20
    input_size = 5
    hidden_size = 16
    
    # Create synthetic time series data
    X_train = np.random.randn(batch_size, seq_len, input_size)
    y_train = np.random.randint(0, 2, (batch_size, 1)).astype(float)
    
    # Build network
    network = NeuralNetwork(name="TestLSTMNetwork")
    network.add_layer(LSTM(input_size, hidden_size, return_sequences=False), name="lstm1")
    network.add_layer(Dense(hidden_size, 1), name="dense1")
    network.add_layer(Sigmoid(), name="sigmoid1")
    
    # Set loss
    loss_fn = BinaryCrossEntropy(from_logits=False)
    network.set_loss(loss_fn)
    
    # Test forward pass
    output = network.forward(X_train)
    print(f"✓ Network forward pass - output shape: {output.shape}")
    assert output.shape == (batch_size, 1), "Network output shape mismatch!"
    
    # Test loss computation
    loss = network.compute_loss(output, y_train)
    print(f"✓ Loss computation - loss: {loss:.4f}")
    assert loss > 0, "Loss should be positive!"
    
    # Test backward pass
    loss_grad = loss_fn.backward()
    network.backward(loss_grad)
    
    # Check gradients
    grads = network.get_grads()
    print(f"✓ Gradients computed for {len(grads)} layers")
    assert len(grads) > 0, "Gradients should be computed!"
    
    # Test with optimizer
    optimizer = Adam(learning_rate=0.001)
    trainer = Trainer(network, optimizer, loss_fn)
    
    # Test training step
    loss, accuracy = trainer.train_step(X_train, y_train)
    print(f"✓ Training step - loss: {loss:.4f}, accuracy: {accuracy:.2f}%")
    
    print("✓ Integration test passed!\n")


def test_conv_lstm_integration():
    """Test Conv1D + LSTM integration."""
    print("=" * 60)
    print("Testing Conv1D + LSTM Integration")
    print("=" * 60)
    
    batch_size = 4
    seq_len = 30
    input_channels = 5
    
    # Build hybrid network
    network = NeuralNetwork(name="ConvLSTM")
    
    # Conv1D layer
    network.add_layer(Conv1D(input_channels, 8, kernel_size=3, padding='same'), name="conv1")
    network.add_layer(ReLU(), name="relu1")
    network.add_layer(MaxPooling1D(pool_size=2, stride=2), name="pool1")
    
    # LSTM layer
    network.add_layer(LSTM(8, 16, return_sequences=False), name="lstm1")
    
    # Dense output
    network.add_layer(Dense(16, 1), name="dense1")
    network.add_layer(Sigmoid(), name="sigmoid1")
    
    # Test forward pass
    X = np.random.randn(batch_size, seq_len, input_channels)
    output = network.forward(X)
    print(f"✓ Conv+LSTM network - input shape: {X.shape}")
    print(f"✓ Conv+LSTM network - output shape: {output.shape}")
    assert output.shape == (batch_size, 1), "Output shape mismatch!"
    
    # Test with loss and backward
    y = np.random.randint(0, 2, (batch_size, 1)).astype(float)
    loss_fn = BinaryCrossEntropy(from_logits=False)
    network.set_loss(loss_fn)
    
    loss = network.compute_loss(output, y)
    loss_grad = loss_fn.backward()
    network.backward(loss_grad)
    
    print(f"✓ Conv+LSTM - loss: {loss:.4f}")
    print("✓ Conv+LSTM integration test passed!\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("KhayatMiniNN - New Components Test Suite")
    print("=" * 60 + "\n")
    
    try:
        test_binary_crossentropy()
        test_lstm()
        test_gru()
        test_conv1d()
        test_pooling()
        test_embedding()
        test_sequence_utils()
        test_integration()
        test_conv_lstm_integration()
        
        print("=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nAll new components are working correctly!")
        print("Your library is ready for time series stock prediction!\n")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ TEST FAILED!")
        print("=" * 60)
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


