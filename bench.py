#!/usr/bin/env python3
import numpy as np

from opt.bench import Bench

def _test_func1(vec):
    m = 1
    for el in vec:
        m *= abs(el)

    return np.sum(abs(vec)) + m

func1 = Bench(dims = 30, up = 10, low = -10, func = _test_func1)
