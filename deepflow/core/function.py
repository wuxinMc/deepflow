from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Function(ABC):
    def __init__(self):
        self.input = None

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        raise NotImplemented

    def __call__(self, *args, **kwargs) -> Any:
        return self.forward(*args, **kwargs)

    @abstractmethod
    def backward(self, *args, **kwargs) -> Any:
        raise NotImplemented


class Add(Function):
    def forward(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        return x1 + x2

    def backward(self, grad_add: np.ndarray) -> Any:
        return grad_add, grad_add


class Sub(Function):
    def forward(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        return x1 - x2

    def backward(self, grad_plus: np.ndarray) -> Any:
        return grad_plus, grad_plus


class Mul(Function):
    def forward(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        return x1 * x2

    def backward(self, grad_mul: np.ndarray) -> Any:
        return grad_mul * self.input[0], grad_mul * self.input[1]


class Div(Function):
    def forward(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        return x1 / x2

    def backward(self, grad_div: np.ndarray) -> Any:
        return grad_div / (self.input[0] ** 2), \
               grad_div / (self.input[1] ** 2)
