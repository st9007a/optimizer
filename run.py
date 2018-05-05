#!/usr/bin/env python3
import numpy as np
from datetime import datetime

from opt.PSO import StandardPSO
from bench import *

if __name__ == '__main__':

    opt = StandardPSO(num_particles = 50, c1 = 0.5, c2 = 0.5)

    start = datetime.now()
    opt.optimize(bench = func6, iters = 1000)
    end = datetime.now()

    print(str(end - start))
    print(opt.best_vec)
    print(func6.eval(opt.best_vec))
