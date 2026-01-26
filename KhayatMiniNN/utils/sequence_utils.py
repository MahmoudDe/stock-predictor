"""
Sequence processing utilities for time series data.
"""

import numpy as np


def create_sequences(data, sequence_length, prediction_horizon=1):
    """
    Create sliding window sequences from time series data.
    
    Args:
        data: (num_samples, num_features) - time series data
        sequence_length: Length of input sequences
        prediction_horizon: How many steps ahead to predict (default: 1)
    
    Returns:
        X: (num_sequences, sequence_length, num_features) - input sequences
        y: (num_sequences, prediction_horizon, num_features) or (num_sequences,) - target sequences
    """
    num_samples, num_features = data.shape
    
    if sequence_length + prediction_horizon > num_samples:
        raise ValueError(
            f"sequence_length ({sequence_length}) + prediction_horizon ({prediction_horizon}) "
            f"exceeds data length ({num_samples})"
        )
    
    num_sequences = num_samples - sequence_length - prediction_horizon + 1
    
    X = np.zeros((num_sequences, sequence_length, num_features))
    y = np.zeros((num_sequences, prediction_horizon, num_features))
    
    for i in range(num_sequences):
        X[i] = data[i:i + sequence_length]
        y[i] = data[i + sequence_length:i + sequence_length + prediction_horizon]
    
    # If prediction_horizon is 1 and single feature, squeeze y
    if prediction_horizon == 1 and num_features == 1:
        y = y.squeeze(axis=(1, 2))
    elif prediction_horizon == 1:
        y = y.squeeze(axis=1)
    
    return X, y


def pad_sequences(sequences, max_length=None, padding='pre', value=0.0):
    """
    Pad variable-length sequences to the same length.
    
    Args:
        sequences: List of arrays or array of variable-length sequences
        max_length: Maximum length (if None, use longest sequence)
        padding: 'pre' or 'post' - where to pad
        value: Value to use for padding
    
    Returns:
        padded: (num_sequences, max_length, num_features) - padded sequences
        lengths: (num_sequences,) - original lengths of sequences
    """
    if isinstance(sequences, list):
        sequences = np.array(sequences, dtype=object)
    
    # Find max length
    if max_length is None:
        max_length = max(len(seq) if seq.ndim == 1 else seq.shape[0] for seq in sequences)
    
    # Get feature dimension
    first_seq = sequences[0] if isinstance(sequences, np.ndarray) and sequences.dtype == object else sequences[0]
    if first_seq.ndim == 1:
        num_features = 1
    else:
        num_features = first_seq.shape[1]
    
    num_sequences = len(sequences)
    lengths = np.zeros(num_sequences, dtype=int)
    
    padded = np.full((num_sequences, max_length, num_features), value, dtype=np.float32)
    
    for i, seq in enumerate(sequences):
        if seq.ndim == 1:
            seq = seq.reshape(-1, 1)
        
        seq_len = seq.shape[0]
        lengths[i] = seq_len
        
        if padding == 'pre':
            padded[i, max_length - seq_len:, :] = seq
        else:  # 'post'
            padded[i, :seq_len, :] = seq
    
    return padded, lengths


def truncate_sequences(sequences, max_length):
    """
    Truncate sequences to a maximum length.
    
    Args:
        sequences: (num_sequences, sequence_length, num_features)
        max_length: Maximum length to keep
    
    Returns:
        truncated: (num_sequences, max_length, num_features)
    """
    if sequences.shape[1] <= max_length:
        return sequences
    
    return sequences[:, :max_length, :]


def split_time_series(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split time series data into train/validation/test sets.
    Maintains temporal order (no shuffling).
    
    Args:
        data: (num_samples, num_features) - time series data
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
    
    Returns:
        train_data, val_data, test_data: Split datasets
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    
    num_samples = data.shape[0]
    
    train_end = int(num_samples * train_ratio)
    val_end = train_end + int(num_samples * val_ratio)
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    return train_data, val_data, test_data


def create_lagged_features(data, lags):
    """
    Create lagged features from time series data.
    
    Args:
        data: (num_samples, num_features) - time series data
        lags: List of lag values (e.g., [1, 2, 3] for 1, 2, 3 steps back)
    
    Returns:
        lagged_data: (num_samples, num_features * (len(lags) + 1)) - original + lagged features
    """
    num_samples, num_features = data.shape
    max_lag = max(lags)
    
    if max_lag >= num_samples:
        raise ValueError(f"max_lag ({max_lag}) must be less than num_samples ({num_samples})")
    
    # Start from max_lag to have all lags available
    num_output_samples = num_samples - max_lag
    num_output_features = num_features * (len(lags) + 1)  # +1 for original
    
    lagged_data = np.zeros((num_output_samples, num_output_features))
    
    for i in range(num_output_samples):
        idx = i + max_lag
        # Original features
        lagged_data[i, :num_features] = data[idx]
        
        # Lagged features
        for j, lag in enumerate(lags):
            start_idx = num_features * (j + 1)
            end_idx = start_idx + num_features
            lagged_data[i, start_idx:end_idx] = data[idx - lag]
    
    return lagged_data


def create_rolling_features(data, windows, functions=['mean', 'std']):
    """
    Create rolling window statistics.
    
    Args:
        data: (num_samples, num_features) - time series data
        windows: List of window sizes (e.g., [7, 30] for 7-day and 30-day windows)
        functions: List of functions to apply ['mean', 'std', 'min', 'max']
    
    Returns:
        rolling_data: (num_samples, num_features * num_windows * num_functions) - rolling features
    """
    num_samples, num_features = data.shape
    max_window = max(windows)
    
    if max_window >= num_samples:
        raise ValueError(f"max_window ({max_window}) must be less than num_samples ({num_samples})")
    
    num_output_features = num_features * len(windows) * len(functions)
    rolling_data = np.zeros((num_samples, num_output_features))
    
    func_map = {
        'mean': np.mean,
        'std': np.std,
        'min': np.min,
        'max': np.max
    }
    
    for i in range(num_samples):
        feature_idx = 0
        
        for window in windows:
            start_idx = max(0, i - window + 1)
            window_data = data[start_idx:i + 1]
            
            for func_name in functions:
                func = func_map[func_name]
                values = func(window_data, axis=0)
                rolling_data[i, feature_idx:feature_idx + num_features] = values
                feature_idx += num_features
    
    return rolling_data


