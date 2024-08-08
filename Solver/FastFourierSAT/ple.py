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
from ..utils.hybrid_loader import Formula

jax.config.update("jax_compilation_cache_dir", "tmp/jax-cache")

def preprocess_xor(xor_clauses):
    max_len = len(max(xor_clauses, key=len))
    addr = np.zeros((len(xor_clauses), max_len))
    sign = np.ones((len(xor_clauses), max_len))
    for i in range(len(xor_clauses)):
        for j in range(len(xor_clauses[i])):
            addr[i, j] = abs(xor_clauses[i][j])
            if xor_clauses[i][j] < 0:
                sign[i, j] = -1
    return jnp.array(addr, dtype = int), jnp.array(sign, dtype=int)

@jax.jit
def fval(x, addr, sign):
    x.at[0].set(1)
    value = sign * x[addr]
    cost = jnp.prod(value, axis = 1)
    return jnp.sum(cost)

@jax.jit
def verify(x, addr, sign):
    x.at[0].set(1)
    value = sign * x[addr]
    return jnp.prod(value, axis = 1) > 0

def solve_parity(filepath, tasks):
    F = Formula()
    F.read_DIMACS(filepath)
    nv = F._n_var
    err = len(F._xor_claus)
    tol = len(F._xor_claus) / 4
    addr, sign = preprocess_xor(F._xor_claus)

    pg = ProjectedGradient(fun=fval, projection=projection_box, maxiter=100, tol=1e-12)

    def opt(x0, addr, sign):
        x0 = pg.run(x0, hyperparams_proj=(-1,1), addr=addr, sign=sign).params
        return x0, verify(x0, addr, sign)
    p_opt = jax.jit(jax.vmap(opt, in_axes=(0, None, None)))

    tic = 0
    err = nv
    while (err > tol):
        key = jax.random.PRNGKey(tic)
        x0 = jax.random.truncated_normal(key, -1.0, 1.0, shape=(tasks, nv))
        x0, res = p_opt(x0, addr, sign)
        if res.sum(axis=1).min() < err:
            err = res.sum(axis=1).min()
            print("o {}:{}".format(tic+1, err))
        tic += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type = str)
    args = parser.parse_args()

    if not args.filepath:
        print("File not found")
        exit(0)

    solve_parity(args.filepath, 4096)

if __name__ == "__main__":
    main()