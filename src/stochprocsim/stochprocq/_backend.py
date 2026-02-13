import numpy as np
import torch

from abc import ABC, abstractmethod

DEFAULT_BACKEND = 'numpy'
BACKENDS = {
    'numpy': np,
    'torch': torch
}
TENSOR_BACKEND = BACKENDS[DEFAULT_BACKEND]

def set_backend(backend: str):
    """
    Set the backend for the library.
    """
    if backend not in BACKENDS:
        raise ValueError(f"Backend {backend} is not supported. " +
                         "Supported backends are: {', '.join(BACKENDS.keys())}")
    global TENSOR_BACKEND
    TENSOR_BACKEND = BACKENDS[backend]
    print(f"Backend set to {backend}.")

class BackendTensor(ABC):
    """
    A wrapper class for tensors that allows switching between different backends.
    """

    @abstractmethod
    def cast(self, x, dtype=None, copy=False):  # pragma: no cover
        """Cast an object as the array type of the current backend.

        Args:
            x: Object to cast to array.
            copy (bool): If ``True`` a copy of the object is created in memory.
        """
        raise NotImplementedError

class NumpyTensor(BackendTensor):
    """
    A wrapper class for numpy tensors.
    """
    def __init__(self, data):
        self.data = np.ndarray(data)
        self.dtype = self.data.dtype
        self.tensor_types = np.ndarray

    def cast(self, x, dtype=None, copy=False):
        if dtype is None:
            dtype = self.dtype
        if isinstance(x, self.tensor_types):
            return x.astype(dtype, copy=copy)
        return np.asarray(x, dtype=dtype, copy=copy if copy else None)
