import numpy as np
import math


def a1():
    a = np.zeros(10, dtype=int)
    np.put(a, 4, 1)
    print(a.tolist())


def b1():
    a = np.arange(10, 50)
    print(a.tolist())


def c1():
    a = np.arange(10, 50)[::-1]
    print(a.tolist())


def d1():
    x = np.arange(16).reshape(4, 4)
    print(x.tolist())


def e1():
    x = np.random.rand(8, 8)
    x_min, x_max = x.min(), x.max()
    print("min: {}, max: {}".format(x_min, x_max))


def f1():
    x_4_3 = np.arange(12).reshape(4, 3)
    x_3_2 = np.arange(6).reshape(3, 2)
    x = np.matmul(x_4_3, x_3_2)
    print(x)


def g1():
    a = np.arange(21)
    a[(8 <= a) & (a <= 16)] *= -1
    print(a)


def h1():
    a = np.arange(20)
    a_sum = np.add.reduce(a)
    print(a_sum)


def i1():
    a = np.arange(25).reshape(5, 5)
    odd = a[::2]
    even = a[:, 1::2]
    print("event: {}\nodd: {}".format(even, odd))


def k1():
    a = np.random.random((10, 2))
    x, y = a[:, 0], a[:, 1]
    r = np.sqrt(x ** 2 + y ** 2)
    t = np.arctan2(y, x)
    print(r)
    print(t)


def l1():
    def scalar(v_0, v_1):
        return v_0 @ v_1

    def magnitude(vec):
        a = 0
        for x in vec:
            a += x ** 2
        return math.sqrt(a)

    v1 = np.arange(5)
    v2 = np.array([-1, 9, 5, 3, 1])
    print("dot: {}, mag: {}".format(scalar(v1, v2), magnitude(v1)))


a1()
b1()
c1()
d1()
e1()
f1()
g1()
h1()
i1()
k1()
l1()
