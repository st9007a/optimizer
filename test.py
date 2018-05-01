#!/usr/bin/env python3
import numpy as np

from PSO import StandardPSO

def test_func(vec):
    print(vec)
    return abs(np.sum(vec))

if __name__ == '__main__':

    opt = StandardPSO(dims = 3, particles = 50, up = 5, low = -5, c1 = 0.5, c2 = 0.5, target_func = test_func)

    for i in range(100):
        opt.step()

    print(opt.gbest_val)
    print(opt.gbest_vec)
