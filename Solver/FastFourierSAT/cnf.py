import math
import os
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_triton_softmax_fusion=true '
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_async_collectives=true '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
    '--xla_gpu_enable_highest_priority_async_stream=true '
)

import jax
import jax.numpy as jnp
import numpy as np
from jaxopt import ProjectedGradient
from jaxopt.projection import projection_box

import argparse
from time import time
from ..utils.cnf_loader import Formula

jax.config.update("jax_compilation_cache_dir", "tmp/jax-cache")

def preprocess(clauses):
    max_len = len(max(clauses, key=len))
    addr = np.zeros((len(clauses), max_len))
    sign = np.ones((len(clauses), max_len))
    for i in range(len(clauses)):
        for j in range(len(clauses[i])):
            addr[i, j] = abs(clauses[i][j])
            if clauses[i][j] < 0:
                sign[i, j] = -1
    return jnp.array(addr, dtype = int), jnp.array(sign, dtype=int), jnp.ones(len(clauses))

@jax.jit
def fval(x, addr, sign, weight):
    x = x.at[0].set(1)
    value = (sign * x[addr] + 1) / 2        # rescale to [0, 1]
    cost = weight * jnp.prod(value, axis = 1)
    return cost.sum()

@jax.jit
def verify(x, addr, sign):
    x = x.at[0].set(1)                      # set x[0] to false
    x = jnp.sign(x) + 0                     # binarize the variables
    value = sign * x[addr]
    return (jnp.min(value, axis = 1) > 0) + 0

def solve_cnf(filepath, tasks, t0):
    F = Formula()
    F.read_DIMACS(filepath)
    nv = F._n_var
    addr, sign, weight = preprocess(F._cnf)

    if len(F._cnf) > 100000:
        return False

    pg = ProjectedGradient(fun=fval, projection=projection_box)

    def opt(x0, addr, sign, weight):
        x0 = pg.run(x0, hyperparams_proj=(-1,1), addr=addr, sign=sign, weight=weight).params
        return x0, verify(x0, addr, sign)
    p_opt = jax.jit(jax.vmap(opt, in_axes=(0, None, None, None)))

    tic = 0
    key = jax.random.PRNGKey(tic)
    x0 = jax.random.truncated_normal(key, -1.0, 1.0, shape=(tasks, nv+1))
    while (time() - t0 < 1000):
        x0, res = p_opt(x0, addr, sign, weight)
        err = res.sum(axis=1).min()
        print("o {}".format(err))
        reset = 1000
        tic += 1
        if tic % reset == 0:
            print("c Reset")
            key = jax.random.PRNGKey(tic)
            x0 = jax.random.truncated_normal(key, -1.0, 1.0, shape=(tasks, nv+1))
            weight = jnp.ones(len(F._cnf))
        else:
            reward = res.sum(axis=0)
            weight = 0.9 * weight + 0.1 * reward/reward.max()

        if err < 0.9:
            return True

    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type = str)
    args = parser.parse_args()

    if not args.filepath:
        print("File not found")
        exit(0)

    if not solve_cnf(args.filepath, 32, time()):
        print("Fails to solve")
        exit(0)

if __name__ == "__main__":
    main()
