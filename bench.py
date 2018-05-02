#!/usr/bin/env python3
from opt.bench import Bench
import math

def _test_func1(vec):
    m = 1
    for el in vec:
        m *= abs(el)

    return sum(abs(vec)) + m

def _test_func2(vec):
    total = 0

    for i in range(len(vec)):
        tmp = 0
        for j in range(i):
            tmp += vec[j]

        total += (tmp ** 2)

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
            return k * ((x - a) ** m)

        elif x >= -a and x <= a:
            return 0

        else:
            return k * ((-x - a) ** m)

    tmp1 = 0
    tmp2 = 0

    for i in range(len(vec) - 1):
        tmp1 += ((y(vec[i]) - 1) ** 2) * (1 + 10 * (math.sin(math.pi * y(vec[i + 1])) ** 2))
        tmp2 += u(vec[i], 10, 100, 4)

    tmp2 += u(vec[-1], 10, 100, 4)

    return math.pi / len(vec) * (tmp1 + y(vec[-1] - 1) ** 2 + 10 * (math.sin(math.pi * y(vec[0])) ** 2)) + tmp2


func1 = Bench(dims = 30, up = 10, low = -10, func = _test_func1)
func2 = Bench(dims = 30, up = 100, low = -100, func = _test_func2)
func3 = Bench(dims = 30, up = 600, low = -600, func = _test_func3)
func4 = Bench(dims = 30, up = 50, low = -50, func = _test_func4)
