#!/usr/bin/env python3

class Bench():

    def __init__(self, dims, func, up, low):
        self.func = func
        self.dims = dims
        self.up = up
        self.low = low

    def eval(self, vector):
        if vector.shape[0] != self.dims:
            raise Exception('Dimension not match.')

        return self.func(vector)
