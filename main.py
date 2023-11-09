import numpy as np
from deepflow.core.variable import Variable
from deepflow import square, \
    numerical_diff, exp, sqrt


if __name__ == '__main__':
    array = np.array([
        [0.5, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ])
    variable = Variable(array, requires_grad=True)

    def func(value):
        value = square(value)
        value = exp(value)
        value = sqrt(value)

        return value

    dydv1 = numerical_diff(func, variable, 1e-5)
    y = func(variable)
    y.backward()
    dydv2 = variable.grad

    print(dydv1, dydv2)
