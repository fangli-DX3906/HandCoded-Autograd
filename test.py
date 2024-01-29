import numpy as np

from core import Tensor
from core import no_grad, Config
from function import broadcast_to, matmul

# testing computational graph
'''
# x = Tensor(2.)
# y = x + x
# y.backward()
# print(x.grad)
#
# x.zero_grad()
# z = x + x + x
# z.backward()
# print(x.grad)
'''

# testing changing mode
'''
# print(Config.no_grad)
# with no_grad():
#     print(Config.no_grad)
#     x = Tensor(2.)
#     a = x ** 2
#     y = a + a
#     y.backward()

# print('*' * 10, 'exiting ', '*' * 10)
# print(Config.no_grad)
# x = Tensor(2.)
# a = x ** 2
# y = a + a
# y.backward()
# print(x.grad)
'''

# testing operation override
'''
a = Tensor(3.)
b = Tensor(2.)
# c = Tensor(1.)

# y = -a
# print(y)
# y.backward()
# print(a.grad)
# print(b.grad)

# y = -a
# print(y)
# y.backward()
# print(a.grad)

# y = a - b
# print(y)
# y.backward()
# print(a.grad)
# print(b.grad)

# y = a / b
# print(y)
# y.backward()
# print(a.grad)
# print(b.grad)
'''

# testing higher order gradient
'''
# y = b ** 3
# y.backward(create_graph=False)
# print('first order gradient: ', b.grad)
# gx = b.grad
# b.zero_grad()
# gx.backward()
# print('second order gradient: ', b.grad)

# a.zero_grad()
# b.zero_grad()
# y = a ** 2 * b ** 2
# y.backward(create_graph=True)
# print(a.grad)
# print(b.grad)
# ag = a.grad
# bg = b.grad
# a.zero_grad()
# b.zero_grad()
# ag.backward()
# print(a.grad)
# a.zero_grad()
# b.zero_grad()
# bg.backward()
# print(b.grad)
'''

# testing shape-related operations
'''
x = Tensor([1, 2, 3])
y = Tensor([[10, 20, 30], [40, 50, 60]])
z = x + y
print(z)
z.backward()
print(x.grad)
'''

# testing matrix multiplication
x = Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
w = Tensor(np.array([[1, 1], [1, 1], [1, 1]]))

y = matmul(x, w)
y.backward()
print(x.grad)
print(w.grad)
