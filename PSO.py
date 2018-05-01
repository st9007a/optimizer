#!/usr/bin/env python3
import numpy as np
from random import uniform
from math import inf

class StandardPSO():

    def __init__(self, dims, particles, up, low, c1, c2, target_func):

        self.dims = dims
        self.num_particles = particles
        self.c1 = c1
        self.c2 = c2
        self.up = up
        self.low = low
        self.target_func = target_func

        mean = (up + low) / 2
        std = (up - low) / 6 # 99.5% confidence interval
        self.particles = np.random.normal(mean, std, [particles, dims])

        # truncated
        self.particles = np.where(self.particles >= up, mean, self.particles)
        self.particles = np.where(self.particles <= low, mean, self.particles)

        self.speed = np.random.normal(0, 1, [particles, dims])

        self.pbest_val = np.zeros([particles])
        self.pbest_val.fill(inf)

        self.pbest_vec = np.zeros([particles, dims])

        self.gbest_val = inf
        self.gbest_vec = np.zeros([dims])

        self.compute()

    def compute(self):
        for idx, vec in enumerate(self.particles):
            eval_val = self.target_func(vec)

            if eval_val < self.pbest_val[idx]:
                self.pbest_val[idx] = eval_val
                self.pbest_vec[idx] = vec

            if eval_val < self.gbest_val:
                self.gbest_val = eval_val
                self.gbest_vec = vec

    def clip(self):
        self.particles = np.where(self.particles >= self.up, self.up - 0.0001, self.particles)
        self.particles = np.where(self.particles <= self.low, self.low + 0.0001, self.particles)

    def step(self):

        for idx, speed in enumerate(self.speed):
            r1 = uniform(0, 1)
            r2 = uniform(0, 1)

            self.speed[idx] = speed + self.c1 * r1 * (self.pbest_vec[idx] - self.particles[idx]) + self.c2 * r2 * (self.gbest_vec - self.particles[idx])
            self.particles[idx] += self.speed[idx]

            self.clip()

        self.compute()
