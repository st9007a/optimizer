#!/usr/bin/env python3
from opt.bench import Bench

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

func1 = Bench(dims = 30, up = 10, low = -10, func = _test_func1)
func2 = Bench(dims = 30, up = 100, low = -100, func = _test_func2)
