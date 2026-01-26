"""
Utility functions for KhayatMiniNN.
"""

import numpy as np

# Import core utilities from parent utils.py
import importlib.util
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
utils_py_path = os.path.join(parent_dir, 'utils.py')
spec = importlib.util.spec_from_file_location("utils_module", utils_py_path)
utils_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils_module)

# Re-export core utilities
softmax = utils_module.softmax
one_hot_encode = utils_module.one_hot_encode
initialize_weights = utils_module.initialize_weights

# Import sequence utilities
from .sequence_utils import (
    create_sequences,
    pad_sequences,
    truncate_sequences,
    split_time_series,
    create_lagged_features,
    create_rolling_features
)

__all__ = [
    "softmax",
    "one_hot_encode",
    "initialize_weights",
    "create_sequences",
    "pad_sequences",
    "truncate_sequences",
    "split_time_series",
    "create_lagged_features",
    "create_rolling_features"
]

