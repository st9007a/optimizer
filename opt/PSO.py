#!/usr/bin/env python3
import numpy as np
from random import uniform
from math import inf

class StandardPSO():

    def __init__(self, num_particles, c1, c2):

        self.num_particles = num_particles
        self.c1 = c1
        self.c2 = c2

    @property
    def best_val(self):
        return self.gbest_val

    @property
    def best_vec(self):
        return self.gbest_vec

    def init(self, bench):
        mean = (bench.up + bench.low) / 2
        std = (bench.up - bench.low) / 6
        self.particles = np.random.normal(mean, std, [self.num_particles, bench.dims])

        # truncated
        self.particles = np.where(self.particles >= bench.up, mean, self.particles)
        self.particles = np.where(self.particles <= bench.low, mean, self.particles)

        self.speed = np.random.normal(0, 1, [self.num_particles, bench.dims])

        self.pbest_val = np.zeros([self.num_particles])
        self.pbest_val.fill(inf)

        self.pbest_vec = np.zeros([self.num_particles, bench.dims])

        self.gbest_val = inf
        self.gbest_vec = np.zeros([bench.dims])

        self.compute(bench)

    def compute(self, bench):
        for idx, vec in enumerate(self.particles):
            eval_val = bench.eval(vec)

            if eval_val < self.pbest_val[idx]:
                self.pbest_val[idx] = eval_val
                self.pbest_vec[idx] = vec

            if eval_val < self.gbest_val:
                self.gbest_val = eval_val
                self.gbest_vec = vec

    def clip(self, bench):
        self.particles = np.where(self.particles >= bench.up, bench.up - 0.0001, self.particles)
        self.particles = np.where(self.particles <= bench.low, bench.low + 0.0001, self.particles)

    def compute_speed(self, speed, idx, iters, r1, r2):
        return speed + self.c1 * r1 * (self.pbest_vec[idx] - self.particles[idx]) + self.c2 * r2 * (self.gbest_vec - self.particles[idx])

    def step(self, iters, bench):

        for idx, speed in enumerate(self.speed):
            r1 = uniform(0, 1)
            r2 = uniform(0, 1)
            self.speed[idx] = self.compute_speed(speed, idx, iters, r1, r2)
            self.particles[idx] += self.speed[idx]

            self.clip(bench)

        self.compute(bench)

    def optimize(self, bench, iters):

        self.init(bench)

        for i in range(iters):
            self.step(i, bench)
