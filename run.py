#!/usr/bin/env python3
import numpy as np

from opt.PSO import StandardPSO
from bench import *

if __name__ == '__main__':

    opt = StandardPSO(num_particles = 50, c1 = 0.5, c2 = 0.5)
    opt.optimize(bench = func2, iters = 500)

    print(opt.best_val)
    print(opt.best_vec)
