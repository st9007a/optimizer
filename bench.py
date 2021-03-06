#!/usr/bin/env python3
from opt.bench import Bench
import math

def _test_func1(vec):
    m = 1
    s = 0
    for el in vec:
        m *= abs(el)
        s += abs(el)

    return s + m

def _test_func2(vec):
    total = 0

    for i in range(len(vec)):
        total += sum(vec[:i]) ** 2

    return total

def _test_func3(vec):
    tmp1 = 0
    tmp2 = 1

    for i, v in enumerate(vec):
        tmp1 += (v ** 2)
        tmp2 *= math.cos(v / math.sqrt(i + 1))

    return tmp1 / 4000 - tmp2 + 1

def _test_func4(vec):
    def y(x):
        return 1 + (x + 1) / 4

    def u(x, a, k, m):
        if x > a:
            return k * (x - a) ** m

        elif x >= -a and x <= a:
            return 0

        else:
            return k * (-x - a) ** m

    tmp1 = 0
    tmp2 = 0

    for i in range(len(vec) - 1):
        tmp1 += (y(vec[i]) - 1) ** 2 * (1 + 10 * math.sin(math.pi * y(vec[i + 1])) ** 2)
        tmp2 += u(vec[i], 10, 100, 4)

    tmp2 += u(vec[-1], 10, 100, 4)

    return math.pi / len(vec) * (tmp1 + y(vec[-1] - 1) ** 2 + 10 * math.sin(math.pi * y(vec[0])) ** 2) + tmp2

def _test_func5(vec):
    def u(x, a, k, m):
        if x > a:
            return k * (x - a) ** m

        elif x >= -a and x <= a:
            return 0

        else:
            return k * (-x - a) ** m

    tmp1 = 0
    tmp2 = 0

    for i in range(len(vec) - 1):
        tmp1 += (vec[i] - 1) ** 2 * (1 + math.sin(3 * math.pi * vec[i + 1]) ** 2)
        tmp2 += u(vec[i], 5, 100, 4)

    tmp2 += u(vec[-1], 5, 100, 4)

    return 0.1 * (math.sin(3 * math.pi * vec[0]) + tmp1 + (vec[-1] - 1) ** 2 * (1 + math.sin(2 * math.pi * vec[-1]) ** 2)) + tmp2

def _test_func6(vec):

    a = [
        [-32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32],
        [-32, -32, -32, -32, -32, -16, -16, -16, -16, -16, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32],
    ]

    tmp = 1 / 500

    for j in range (25):
        s = 0

        for i in range(2):
            s += (vec[i] - a[i][j]) ** 6

        tmp += 1 / (j + 1 + s)

    return tmp ** (-1)

def _test_func7(vec):

    a = [0.1957, 0.1947, 0.1735, 0.1600, 0.0844, 0.0627, 0.0456, 0.0342, 0.0342, 0.0235, 0.0246]
    b = [0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16]

    tmp = 0
    for ai, bi in zip(a, b):
        tmp += (ai - vec[0] * (bi ** 2 + bi * vec[1]) / bi ** 2 + bi * vec[2] + vec[3]) ** 2

    return tmp

def _test_func8(vec):
    return 4 * vec[0] ** 2 - 2.1 * vec[0] ** 4 + 1 / 3 * vec[0] ** 6 + vec[0] * vec[1] - 4 * vec[1] ** 2 + 4 * vec[1] ** 4


func1 = Bench(dims = 30, up = 10, low = -10, func = _test_func1, optima = 0)
func2 = Bench(dims = 30, up = 100, low = -100, func = _test_func2, optima = 0)
func3 = Bench(dims = 30, up = 600, low = -600, func = _test_func3, optima = 0)
func4 = Bench(dims = 30, up = 50, low = -50, func = _test_func4, optima = 0)
func5 = Bench(dims = 30, up = 50, low = -50, func = _test_func5, optima = 0)
func6 = Bench(dims = 2, up = 65, low = -65, func = _test_func6, optima = 1)
func7 = Bench(dims = 4, up = 5, low = -5, func = _test_func7, optima = 0.0003)
func8 = Bench(dims = 2, up = 5, low = -5, func = _test_func8, optima = -1.0316)
