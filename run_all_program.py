#!/usr/bin/env python3
import pickle
import numpy as np
from datetime import datetime
from pprint import pprint

from opt.GA import GA
from opt.GSA import GSA
from opt.PSO import StandardPSO
from bench import *

def run_program(optimizer, bench, iters, path):
    mean_fitness_last_iter = []
    history_100_rounds = []

    print('Start ' + path)

    for i in range(100):
        print('Round ' + str(i + 1))
        optimizer.optimize(bench = bench, iters = iters)

        history_100_rounds.append(optimizer.history)
        mean_fitness_last_iter.append(optimizer.mean_fitness_last_iter(bench))

    res = {
        'history': history_100_rounds,
        'mean_fitness_last_iter': mean_fitness_last_iter,
    }

    with open(path, 'wb') as p:
        pickle.dump(res, p, protocol = pickle.HIGHEST_PROTOCOL)

    print('Done, save result to ' + path)

if __name__ == '__main__':

    benchs = [func1, func2, func3, func4, func5, func6, func7, func8]
    GAOpitimizer = GA(num_population = 50, crossover_p = 0.8, mutation_p = 0.008)
    GSAOptimizer = GSA(num_agents = 50, g = 9.8, kbest = 40, kbest_decay = 3e-3, g_decay = 5e-3)
    PSOOptimizer = StandardPSO(c1 = 0.5, c2 = 0.5, num_particles = 50)

    for idx, b in enumerate(benchs):
        run_program(GAOpitimizer, b, 1000, 'results/GA_func%d.pkl' % (idx + 1))
        run_program(GSAOptimizer, b, 1000, 'results/GSA_func%d.pkl' % (idx + 1))
        run_program(PSOOptimizer, b, 1000, 'results/PSO_func%d.pkl' % (idx + 1))
