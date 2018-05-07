#!/usr/bin/env python3
import numpy as np
from datetime import datetime

from opt.GSA import GSA
from bench import *

if __name__ == '__main__':

    opt = GSA(num_agents = 50, g = 9.8, kbest = 40)

    start = datetime.now()
    opt.optimize(bench = func8, iters = 1000)
    end = datetime.now()

    print(str(end - start))
    print(opt.best_vec)
    print(func8.eval(opt.best_vec))
