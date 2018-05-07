#!/usr/bin/env python3
import numpy as np

from random import uniform
from math import inf, log
from heapq import nlargest

class GSA():

    def __init__(self, num_agents, g, kbest, kbest_decay = 1e-2, g_decay = 1e-2, epsilon = 1e-3):
        self.g = g
        self.g_decay = g_decay
        self.kbest = kbest
        self.kbest_decay = kbest_decay
        self.epsilon = epsilon
        self.num_agents = num_agents

        self.mass_score = [0] * num_agents
        self.mass = [0] * num_agents

        self.best_vec = None
        self.best_score = -inf

    def init(self, bench):
        mean = (bench.up + bench.low) / 2
        std = (bench.up - bench.low) / 6
        self.particles = np.random.normal(mean, std, [self.num_agents, bench.dims])

        # truncated
        np.clip(self.particles, bench.low, bench.up, out = self.particles)

        self.speed = np.random.normal(0, 1, [self.num_agents, bench.dims])

    def mass_fitness_score(self, bench, vec):
        score = 1 / (abs(bench.eval(list(vec)) - bench.optima) + 1e-4)
        # score = -log(abs(bench.eval(list(vec)) - bench.optima) + 1e-4)

        if score > self.best_score:
            self.best_score = score
            self.best_vec = list(vec)

        return score

    def mass_param(self):

        best = max(self.mass_score)
        worst = min(self.mass_score)

        mass = 0

        for f in self.mass_score:
            mass += (f - worst + 1e-4) / (best - worst + 1e-4)

        return mass, best, worst

    def clip(self, bench):
        self.particles = np.where(self.particles >= bench.up, bench.up - 0.0001, self.particles)
        self.particles = np.where(self.particles <= bench.low, bench.low + 0.0001, self.particles)

    def step(self, iters, bench):

        # Evaluate fitness score
        for idx, pos in enumerate(self.particles):
            self.mass_score[idx] = self.mass_fitness_score(bench, pos)

        # Update G, kbest, best, worst
        g = self.g * (1 - self.g_decay) ** iters
        kbest = int(self.kbest * (1 - self.kbest_decay) ** iters)
        mass_sum, best, worst = self.mass_param()

        # Update mass
        for i in range(self.num_agents):
            self.mass[i] = (self.mass_score[i] - worst + 1e-4) / (best - worst + 1e-4) / mass_sum

        # Calcuate a
        kbest_mass = nlargest(kbest, self.mass)

        for i in range(self.num_agents):
            f = np.zeros([bench.dims])
            ri = uniform(0, 1)

            for m in kbest_mass:
                j = self.mass.index(m)

                if i == j:
                    continue

                rj = uniform(0, 1)
                f += rj * g * self.mass[i] * self.mass[j] / np.linalg.norm(self.particles[j] - self.particles[i]) * (self.particles[j] - self.particles[i])

            a = f / self.mass[i]

            self.speed[i] = ri * self.speed[i] + a


        for i in range(self.num_agents):
            self.particles[i] += self.speed[i]

        np.clip(self.particles, bench.low, bench.up, out = self.particles)

    def optimize(self, bench, iters):

        self.init(bench)

        for i in range(iters):
            self.step(i, bench)
