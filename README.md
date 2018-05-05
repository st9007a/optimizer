# Optimizer

## Description

find a input vector X = [x1, x2, ..., xn] let the function f(X) = **optima value**

## Usage

1. define the function your want to optimize:

```python3
from opt.bench import Bench


# Input:
#   vec(List): 1-D List which represent input vector [x1, x2, ... , xn]
# Output:
#   val(Float): a single value which is the output of this function
def my_func(vec):
    m = 1
    s = 0
    for el in vec:
        m *= abs(el)
        s += abs(el)

    return s + m

bench_func = Bench(
    dims = 30,      # dimension of input vector
    up = 10,        # upper bound of search space
    low = -10,      # lower bound of search space
    func = my_func, # function you want to optimize
    optima = 0,     # output of the function
)
```

2. define the optimizer

```python3
from opt.PSO import StandardPSO

optimizer = StandardPSO(
    num_particles = 50, # total count of particles
    c1 = 0.5,           # coefficient of pbest
    c2 = 0.5,           # coefficient of gbest
)

optimizer.optimize(bench = bench_func, iters = 1000)

# print input vector which optimizer found
print(optimizer.best_vec)

# print output value of the function
print(bench_func.eval(optimizer.best_vec))
```
