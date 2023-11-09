from typing import Callable

import numpy as np
from numpy import ndarray

from deepflow.core.function import Function
from deepflow.core.variable import Variable


class Square(Function):
    def forward(self, x: ndarray) -> ndarray:
        return x ** 2

    def backward(self, grad_y: ndarray) -> ndarray:
        x = self.input.data
        grad_x = grad_y * 2 * x
        return grad_x


class Sqrt(Function):
    def forward(self, x: ndarray) -> ndarray:
        return x ** 0.5

    def backward(self, grad_y: ndarray) -> ndarray:
        x = self.input.data
        grad_x = grad_y / (2 * x ** 0.5)
        return grad_x


class Exp(Function):
    def forward(self, x: ndarray) -> ndarray:
        return np.exp(x)

    def backward(self, grad_y: ndarray) -> ndarray:
        x = self.input.data
        grad_x = grad_y * np.exp(x)
        return grad_x


class Log(Function):
    def forward(self, x: ndarray) -> ndarray:
        return np.log(x)

    def backward(self, grad_y: ndarray) -> ndarray:
        x = self.input.data
        grad_x = grad_y / x
        return grad_x


def square(x: Variable) -> Variable:
    """
    计算平方

    :math:`y = x^2`

    Args:
        x (Variable): 输入向量

    Returns:
        Variable: 输出向量
    """

    return Square()(x)


def sqrt(x: Variable) -> Variable:
    """
    计算平方根

    :math:`y = \\sqrt{x}`

    Args:
        x (Variable): 输入向量

    Returns:
        Variable: 输出向量
    """

    return Sqrt()(x)


def exp(x: Variable) -> Variable:
    """
    计算指数

    :math:`y = e^x`

    Args:
        x (Variable): 输入向量

    Returns:
        Variable: 输出向量
    """

    return Exp()(x)


def log(x: Variable) -> Variable:
    """
    计算对数

    :math:`y = \\ln{x}`

    Args:
        x (Variable): 输入向量

    Returns:
        Variable: 输出向量
    """

    return Log()(x)


def numerical_diff(
        func: Callable,
        x: Variable,
        eps: float = 1e-5
) -> ndarray:
    """
    计算函数在某点的数值导数

    :math:`d_y = \\frac{x_1 - x_2}{2 × eps}`

    Args:
        func (Function): 待求导的函数
        x (Variable): 输入向量
        eps (float): 微小变化量，默认值为 1e-5

    Returns:
        ndarray: 数值导数结果
    """

    x1 = Variable(x.data - eps)
    x2 = Variable(x.data + eps)
    y1 = func(x1)
    y2 = func(x2)

    return (y2.data - y1.data) / (2 * eps)
