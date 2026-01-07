import numpy as np


def softmax(x):
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def one_hot_encode(y, num_classes):
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot


def initialize_weights(shape, method='xavier'):
    if method == 'xavier':
        limit = np.sqrt(6.0 / (shape[0] + shape[1]))
        return np.random.uniform(-limit, limit, shape)
    elif method == 'he':
        return np.random.randn(*shape) * np.sqrt(2.0 / shape[0])
    else:
        return np.random.randn(*shape) * 0.01
