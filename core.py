import numpy as np
import contextlib


# a Tensor class

class Tensor:
    def __init__(self, data):
        self.data = np.atleast_1d(np.asarray(data))
        self.grad = None
        self._generator = None
        self.priority = 0

    @property
    def generator(self):
        return self._generator

    @generator.setter
    def generator(self, func):
        self._generator = func
        self.priority = func.priority + 1

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def __str__(self):
        return f'data:{self.data}, shape={self.data.shape}, grad=({self.grad})'

    def __repr__(self):
        return self.__str__()

    def __add__(self, other):
        pass

    def __mul__(self, other):
        pass

    def __neg__(self):
        pass

    def __sub__(self, other):
        pass

    def __truediv__(self, other):
        pass

    def __pow__(self, power):
        pass

    def zero_grad(self):
        self.grad = None

    def backward(self, retain_graph=False, create_graph=False):
        if self.grad is None:
            self.grad = Tensor(np.ones_like(self.data))

        func_list = []
        func_set = set()

        def append_func(func):
            if func not in func_set:
                func_list.append(func)
                # NoneType has no attribute 'priority', if no computational graph
                func_list.sort(key=lambda x: x.priority)
                func_set.add(func)

        append_func(self.generator)

        while len(func_list):
            generator = func_list.pop()
            if generator is None:
                break

            inputs, outputs = generator.inputs, generator.outputs
            grad_ys = [y().grad for y in outputs]

            if not create_graph:
                Config.no_grad = True

            grad_xs = generator.backward(*grad_ys)
            for x, grad_x in zip(inputs, grad_xs):
                if x.grad is None:
                    x.grad = grad_x
                else:
                    x.grad = x.grad + grad_x
                if x.generator is not None:
                    append_func(x.generator)

            if not retain_graph:
                for o in outputs:
                    o().grad = None

            Config.no_grad = False


# configuration

class Config:
    no_grad = False


@contextlib.contextmanager
def no_grad():
    Config.no_grad = True
    try:
        yield
    except Exception:
        raise Exception('No computational graph.')
    finally:
        Config.no_grad = False
