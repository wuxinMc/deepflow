import typing
from typing import Optional

import numpy as np
from numpy import ndarray
if typing.TYPE_CHECKING:
    from .function import Function


class Variable(object):
    """
    变量类，用于存储张量数据和计算梯度。

    Attributes:
        data (ndarray): 存储的张量数据。
        grad (Optional[ndarray]): 存储变量的梯度，默认为None。
        _requires_grad (bool): 是否需要计算梯度。
        _grad_fn (Optional[Function]): 存储变量的梯度函数，默认为None。

    Methods:
        set_grad_fn(self, grad_fn: Function): 设置变量的梯度函数。
        grad_fn(self): 返回变量的梯度函数。
        backward(self) -> None: 反向传播梯度。

    Raises:
        RuntimeError: 当requires_grad为False时，调用backward()方法会抛出此异常。
    """

    def __init__(
            self,
            data: ndarray,
            requires_grad: bool = False
    ):
        """
        Args:
            data(ndarray): 输入数据
            requires_grad: 是否需要计算梯度，默认为False
        """

        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a numpy.ndarray, get {}.".format(type(data)))

        self.data: np.ndarray = np.atleast_1d(data)
        self.grad: Optional["ndarray"] = None
        self._requires_grad: bool = requires_grad
        self._grad_fn: Optional["Function"] = None

    def set_grad_fn(self, grad_fn: "Function"):
        self._grad_fn = grad_fn

    def set_requires_grad(self, requires_grad: bool):
        self._requires_grad = requires_grad

    @property
    def grad_fn(self):
        return self._grad_fn

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def ndims(self) -> int:
        return len(self.shape)

    def numpy(self) -> ndarray:
        return self.data

    def backward(self) -> None:
        """
        这是一个反向传播函数，用于计算梯度。

        Returns:
            None

        Raises:
            RuntimeError: 如果 `requires_grad` 为 False，则抛出运行时错误

        Examples:
            >>> self.backward()
        """

        if not self.requires_grad:
            raise RuntimeError("Variable does not require grad, Please set `requires_grad` True!")

        grad_funcs = [self.grad_fn]
        while grad_funcs:
            grad_fn = grad_funcs.pop()
            if self.grad is None:
                self.grad = np.ones_like(self.data)
            inputs, outputs = grad_fn.input, grad_fn.output
            inputs.grad = grad_fn.backward(outputs.grad)

            if inputs.grad_fn is not None:
                grad_funcs.append(inputs.grad_fn)
