from abc import ABC, abstractmethod

from numpy import ndarray

from .variable import Variable


class Function(ABC):
    input: Variable
    output: Variable

    @abstractmethod
    def forward(self, x: ndarray) -> ndarray:
        raise NotImplementedError

    def backward(self, grad_input: ndarray) -> ndarray:
        raise NotImplementedError

    def __call__(self, inputs: Variable) -> Variable:
        """
        这是一个call函数，用于执行模型的前向传播过程并生成输出。

        Args:
            inputs(Variable): 输入的数据，包含输入的数据和相关的梯度信息。

        Returns:
            outputs(Variable): 模型的输出，包含模型的预测结果和相关的梯度信息。
        """

        x = inputs.data
        y = self.forward(x)
        outputs = Variable(y)
        if inputs.requires_grad:
            outputs.set_grad_fn(self)
            outputs.set_requires_grad(True)
        self.input = inputs
        self.output = outputs
        return outputs
