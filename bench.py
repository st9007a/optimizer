#!/usr/bin/env python3
from opt.bench import Bench
from decimal import *

getcontext().prec = 100

def _test_func1(vec):
    m = 1
    for el in vec:
        m *= abs(el)

    return sum(abs(vec)) + m

def _test_func2(vec):
    total = Decimal(1)

    vec = [Decimal(v) for v in vec]

    for i in range(len(vec)):
        tmp = Decimal(1)
        for j in range(i):
            tmp *= vec[j]

        tmp *= tmp
        total *= tmp

    return total

func1 = Bench(dims = 30, up = 10, low = -10, func = _test_func1)
func2 = Bench(dims = 10, up = 100, low = -100, func = _test_func2)
