import numpy as np
from typing import Tuple
import weakref

from core import Tensor, Config
from utils import sum_to


# Tensor operations

class Function:
    def __call__(self, *inputs):
        self.inputs = inputs
        outputs = self.forward(*inputs)

        if not Config.no_grad:
            self.outputs = tuple([weakref.ref(o) for o in outputs])
            self.priority = max([x.priority for x in self.inputs])
            for tensor in outputs:
                tensor.generator = self

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, *inputs) -> Tuple[Tensor]:
        raise NotImplementedError

    def backward(self, grad_y) -> Tuple[Tensor, ...]:
        raise NotImplementedError


# operation classes: operation on Tensor.data does not create the computational graph.

class Add(Function):
    def forward(self, x0: Tensor, x1: Tensor) -> Tuple[Tensor]:
        self.x0_shape = x0.shape
        self.x1_shape = x1.shape
        return Tensor(x0.data + x1.data),

    def backward(self, grad_y: Tensor) -> Tuple[Tensor, Tensor]:
        grad_0, grad_1 = grad_y, grad_y
        if self.x1_shape != self.x0_shape:
            grad_0 = sum_to(grad_y, self.x0_shape)
            grad_1 = sum_to(grad_y, self.x1_shape)
        return grad_0, grad_1


class Sub(Function):
    def forward(self, x0: Tensor, x1: Tensor) -> Tuple[Tensor]:
        return Tensor(x0.data - x1.data),

    def backward(self, grad_y: Tensor) -> Tuple[Tensor, Tensor]:
        return grad_y, -grad_y


class Mul(Function):
    def forward(self, x0: Tensor, x1: Tensor) -> Tuple[Tensor]:
        return Tensor(x0.data * x1.data),

    def backward(self, grad_y: Tensor) -> Tuple[Tensor, Tensor]:
        x0 = self.inputs[0]
        x1 = self.inputs[1]
        return x1 * grad_y, x0 * grad_y


class Div(Function):
    def forward(self, x0: Tensor, x1: Tensor) -> Tuple[Tensor]:
        return Tensor(x0.data / x1.data),

    def backward(self, grad_y: Tensor) -> Tuple[Tensor, Tensor]:
        x0 = self.inputs[0]
        x1 = self.inputs[1]
        grad_x0 = grad_y / x1
        grad_x1 = - grad_y * x0 / x1 ** 2
        return grad_x0, grad_x1


class Square(Function):
    def forward(self, x: Tensor) -> Tuple[Tensor]:
        y = x.data ** 2
        return Tensor(y),

    def backward(self, grad_y: Tensor) -> Tuple[Tensor]:
        x = self.inputs[0]
        return grad_y * Tensor(2) * x,


class Exp(Function):
    def forward(self, x: Tensor) -> Tuple[Tensor]:
        y = np.exp(x.data)
        return Tensor(y),

    def backward(self, grad_y: Tensor) -> Tuple[Tensor]:
        x = self.inputs[0]
        return Tensor(np.exp(x.data)) * grad_y,


class Neg(Function):
    def forward(self, x: Tensor) -> Tuple[Tensor]:
        return Tensor(-x.data),

    def backward(self, grad_y: Tensor) -> Tuple[Tensor]:
        return -grad_y,


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        return Tensor(x.data ** self.c),

    def backward(self, grad_y: Tensor) -> Tuple[Tensor]:
        x = self.inputs[0]
        return Tensor(self.c) * x ** (self.c - 1) * grad_y,


class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape
        self.x_shape = None

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        self.x_shape = x.shape
        y = np.reshape(x.data, self.shape)
        return Tensor(y),

    def backward(self, grad_y: Tensor) -> Tuple[Tensor]:
        grad_x = Tensor(np.reshape(grad_y.data, self.x_shape))
        return grad_x,


class Transpose(Function):
    def forward(self, x: Tensor) -> Tuple[Tensor]:
        y = np.transpose(x.data)
        return Tensor(y),

    def backward(self, grad_y: Tensor) -> Tuple[Tensor]:
        grad_x = Tensor(np.transpose(grad_y.data))
        return (grad_x,)


class BroadCastTo(Function):
    def __init__(self, shape: tuple):
        self.shape = shape
        self.x_shape = None

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        self.x_shape = x.shape
        y = np.broadcast_to(x.data, self.shape)
        return Tensor(y),

    def backward(self, grad_y: Tensor) -> Tuple[Tensor]:
        grad_x = sum_to(grad_y, self.x_shape)
        return grad_x,


# matrix operation
class MatMul(Function):
    def forward(self, x: Tensor, w: Tensor) -> Tuple[Tensor]:
        y = np.dot(x.data, w.data)
        return Tensor(y),

    def backward(self, grad_y: Tensor) -> Tuple[Tensor, Tensor]:
        x, w = self.inputs
        grad_x = matmul(grad_y, transpose(w))
        grad_w = matmul(transpose(x), grad_y)
        return grad_x, grad_w


# lower case funcs

def add(x0: Tensor, x1: Tensor):
    return Add()(x0, x1)


def mul(x0: Tensor, x1: Tensor):
    return Mul()(x0, x1)


def square(x: Tensor):
    return Square()(x)


def exp(x: Tensor):
    return Exp()(x)


def neg(x: Tensor):
    return Neg()(x)


def sub(x0: Tensor, x1: Tensor):
    return Sub()(x0, x1)


def div(x0: Tensor, x1: Tensor):
    if x1.data == 0:
        raise ZeroDivisionError
    else:
        return Div()(x0, x1)


def pow(c, x: Tensor):
    return Pow(c)(x)


def reshape(x: Tensor, shape: tuple):
    return Reshape(shape)(x)


def transpose(x: Tensor):
    return Transpose()(x)


def broadcast_to(x: Tensor, shape: tuple):
    if x.shape == shape:
        return x
    else:
        return BroadCastTo(shape)(x)


def matmul(x: Tensor, w: Tensor):
    return MatMul()(x, w)


# define Tensor operations
Tensor.__add__ = add
Tensor.__sub__ = sub
Tensor.__mul__ = mul
Tensor.__truediv__ = div
Tensor.__pow__ = pow
Tensor.__neg__ = neg
