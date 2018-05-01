#!/usr/bin/env python3
import numpy as np

from opt.PSO import StandardPSO

def test_func(vec):
    return abs(np.sum(vec))

if __name__ == '__main__':

    opt = StandardPSO(dims = 10, iters = 200, particles = 50, up = 5, low = -5, c1 = 0.5, c2 = 0.5, target_func = test_func)

    opt.exec()

    print(opt.gbest_val)
    print(opt.gbest_vec)
