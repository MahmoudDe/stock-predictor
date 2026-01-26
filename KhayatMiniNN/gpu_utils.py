"""
GPU Utilities for Neural Network Acceleration

This module provides GPU support using CuPy when available,
with automatic fallback to NumPy on CPU.
"""

import os

# Try to import CuPy for GPU support
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

import numpy as np


class DeviceManager:
    """
    Manages device selection (CPU/GPU) and provides unified interface
    for array operations.
    """
    
    def __init__(self, device='auto'):
        """
        Initialize device manager.
        
        Args:
            device: 'auto', 'gpu', 'cpu', or specific GPU device ID
        """
        self.device_type = None
        self.device_id = None
        self.use_gpu = False
        
        if device == 'auto':
            # Auto-detect: use GPU if CuPy is available
            if CUPY_AVAILABLE:
                try:
                    # Test if GPU is available
                    _ = cp.array([1])
                    self.use_gpu = True
                    self.device_type = 'gpu'
                    self.device_id = 0
                    print("✓ GPU detected and enabled (CuPy)")
                except:
                    self.use_gpu = False
                    self.device_type = 'cpu'
                    print("⚠ CuPy installed but GPU not available, using CPU")
            else:
                self.use_gpu = False
                self.device_type = 'cpu'
                print("ℹ CuPy not available, using CPU (NumPy)")
        elif device == 'gpu':
            if not CUPY_AVAILABLE:
                raise RuntimeError("CuPy not installed. Install with: pip install cupy-cuda11x (or appropriate version)")
            self.use_gpu = True
            self.device_type = 'gpu'
            self.device_id = 0
        elif device == 'cpu':
            self.use_gpu = False
            self.device_type = 'cpu'
        else:
            raise ValueError(f"Unknown device: {device}. Use 'auto', 'gpu', or 'cpu'")
    
    def get_array_module(self):
        """Get the appropriate array module (cupy or numpy)."""
        return cp if self.use_gpu else np
    
    def array(self, obj, dtype=None):
        """Create array on current device."""
        xp = self.get_array_module()
        return xp.array(obj, dtype=dtype)
    
    def asarray(self, obj, dtype=None):
        """Convert to array on current device."""
        xp = self.get_array_module()
        return xp.asarray(obj, dtype=dtype)
    
    def zeros(self, shape, dtype=None):
        """Create zeros array on current device."""
        xp = self.get_array_module()
        return xp.zeros(shape, dtype=dtype)
    
    def zeros_like(self, a, dtype=None):
        """Create zeros array like input on current device."""
        xp = self.get_array_module()
        return xp.zeros_like(a, dtype=dtype)
    
    def ones(self, shape, dtype=None):
        """Create ones array on current device."""
        xp = self.get_array_module()
        return xp.ones(shape, dtype=dtype)
    
    def empty(self, shape, dtype=None):
        """Create empty array on current device."""
        xp = self.get_array_module()
        return xp.empty(shape, dtype=dtype)
    
    def to_numpy(self, arr):
        """Convert array to NumPy (CPU) array."""
        if self.use_gpu and hasattr(arr, 'get'):
            # CuPy array - transfer to CPU
            return arr.get()
        return np.asarray(arr)
    
    def to_device(self, arr):
        """Move array to current device."""
        if self.use_gpu:
            if isinstance(arr, np.ndarray):
                # NumPy array - move to GPU
                return cp.asarray(arr)
            elif hasattr(arr, 'device'):
                # Already on GPU
                return arr
            else:
                return cp.asarray(arr)
        else:
            if hasattr(arr, 'get'):
                # CuPy array - move to CPU
                return arr.get()
            return np.asarray(arr)
    
    def synchronize(self):
        """Synchronize GPU operations (no-op on CPU)."""
        if self.use_gpu:
            cp.cuda.Stream.null.synchronize()


# Global device manager instance
_default_device_manager = None


def get_device_manager(device='auto'):
    """Get or create default device manager."""
    global _default_device_manager
    if _default_device_manager is None:
        _default_device_manager = DeviceManager(device=device)
    return _default_device_manager


def set_device(device='auto'):
    """Set the default device for computations."""
    global _default_device_manager
    _default_device_manager = DeviceManager(device=device)
    return _default_device_manager


def get_xp(device_manager=None):
    """
    Get the appropriate array module (cupy or numpy).
    
    Args:
        device_manager: DeviceManager instance. If None, uses default.
    
    Returns:
        cupy or numpy module
    """
    if device_manager is None:
        device_manager = get_device_manager()
    return device_manager.get_array_module()


def to_numpy(arr, device_manager=None):
    """Convert array to NumPy (CPU) array."""
    if device_manager is None:
        device_manager = get_device_manager()
    return device_manager.to_numpy(arr)


def to_device(arr, device_manager=None):
    """Move array to current device."""
    if device_manager is None:
        device_manager = get_device_manager()
    return device_manager.to_device(arr)
