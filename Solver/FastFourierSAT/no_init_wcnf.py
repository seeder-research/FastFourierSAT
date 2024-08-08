import sys
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
import random
from ..utils.wcnf_loader import Formula

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
    return jnp.array(addr, dtype = int), jnp.array(sign, dtype=int)

@jax.jit
def fval_cnf(x, addr, sign):
    x = x.at[0].set(1)
    value = (sign * x[addr] + 1) / 2        # rescale to [0, 1]
    cost = jnp.prod(value, axis = 1)
    return cost.sum()

@jax.jit
def fval(x, weight, addr_hard, sign_hard, addr_soft, sign_soft):
    return weight * fval_cnf(x, addr_hard, sign_hard) + fval_cnf(x, addr_soft, sign_soft)

@jax.jit
def verify(x, addr, sign):
    x = x.at[0].set(1)                      # set x[0] to false
    value = sign * x[addr]
    return (jnp.min(value, axis = 1) >= 0).sum()

def run(filepath, tasks):
    t0 = time()
    F = Formula()
    F.read_DIMACS(filepath)
    weight = len(F._soft) + 1

    addr_soft, sign_soft = preprocess(F._soft)
    nv = addr_soft.max()
    best_cost = len(F._soft)

    # Partial MaxSAT
    if len(F._hard) > 0:
        addr_hard, sign_hard = preprocess(F._hard)
        if addr_hard.max() > nv:
            nv = addr_hard.max()

        pg = ProjectedGradient(fun=fval, projection=projection_box)
        def opt(x0, weight, addr_hard, sign_hard, addr_soft, sign_soft):
            x0 = pg.run(x0, hyperparams_proj=(-1,1), weight=weight, 
                        addr_hard=addr_hard, sign_hard=sign_hard, 
                        addr_soft=addr_soft, sign_soft=sign_soft).params
            return x0, verify(x0, addr_hard, sign_hard), verify(x0, addr_soft, sign_soft)
        p_opt = jax.jit(jax.vmap(opt, in_axes=(0, None, None, None, None, None)))

        tic = 0
        while (time() - t0 < 300):
            key = jax.random.PRNGKey(tic)
            x0 = jax.random.truncated_normal(key, -1.0, 1.0, shape=(tasks, nv+1))
    
            # Optimization
            x0, cost_hard, cost_soft = p_opt(x0, weight, addr_hard, sign_hard, addr_soft, sign_soft)
            cost = weight * cost_hard + cost_soft
            if cost.min() < best_cost:
                best_cost = cost.min()
                print("{:.2f}/300.00 o {}".format(time()-t0, best_cost))
                sys.stdout.flush()
                best_x = x0[cost.argmin()]
            if best_cost == 0:
                break
            tic += 1
    # MaxSAT
    else:
        print("{:.2f}/300.00 o {}".format(time()-t0, best_cost))
        pg = ProjectedGradient(fun=fval_cnf, projection=projection_box)
        def opt(x0, addr_soft, sign_soft):
            x0 = pg.run(x0, hyperparams_proj=(-1,1), addr=addr_soft, sign=sign_soft).params
            return x0, verify(x0, addr_soft, sign_soft)
        p_opt = jax.jit(jax.vmap(opt, in_axes=(0, None, None)))

        tic = 0
        while (time() - t0 < 300):
            key = jax.random.PRNGKey(tic)
            x0 = jax.random.truncated_normal(key, -1.0, 1.0, shape=(tasks, nv+1))
    
            # Optimization
            x0, cost = p_opt(x0, addr_soft, sign_soft)
            if cost.min() < best_cost:
                best_cost = cost.min()
                print("{:.2f}/300.00 o {}".format(time()-t0, best_cost))
                sys.stdout.flush()
                best_x = x0[cost.argmin()]
            if best_cost == 0:
                break
            tic += 1
    output_string = []
    for i in range(1, nv+1):
        output_string.append(i if best_x[i] < 0 else -i)
    print(' '.join(map(str, output_string)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type = str)
    args = parser.parse_args()

    if not args.filepath:
        print("File not found")
        exit(0)
    
    res = run(args.filepath, 32)

if __name__ == "__main__":
    main()