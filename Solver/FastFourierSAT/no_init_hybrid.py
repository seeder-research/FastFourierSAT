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
from ..utils.whybrid_loader import Formula
from ..utils.coef import EOFC

jax.config.update("jax_enable_x64", True)

def read_hwcnf(path):
    F = Formula()
    F.read_DIMACS(path)
    return F._cnf_claus, F._eo_claus, F._xor_claus, F._n_var

def fft(n):
    coef = jnp.array(EOFC(n))
    omega = jnp.exp(2j * jnp.pi * jnp.arange(n+1) / (n+1)).reshape(n+1, 1)
    inv_dft = omega ** jnp.arange(n+1) / (n+1)
    omega = jnp.exp(-2j * jnp.pi * jnp.arange(len(inv_dft)) / len(inv_dft)).reshape(n+1, 1, 1)
    return jnp.broadcast_to(omega, (n+1,1,1)), coef[::-1]@inv_dft

def preprocess_eo(clauses):
    eo = np.abs(clauses) - 1
    m, n = eo.shape
    return jnp.array(eo, dtype = int), fft(n)

def preprocess_cnf(clauses):
    cnf = np.abs(clauses) - 1
    return jnp.array(cnf, dtype = int)

def preprocess_xor(clauses):
    xor = np.abs(clauses) - 1
    return jnp.array(xor, dtype = int)

@jax.jit
def fval_eo(x, addr, dft, idft):
    prod = jnp.prod(dft+x[addr], axis=2)
    cost = (idft@prod).real
    return jnp.sum(cost)

@jax.jit
def fval_cnf(x, addr):
    value = -x[addr]            # all negative literals 
    cost = jnp.prod((value+1)/2, axis=1) * 2 - 1
    return jnp.sum(cost)

@jax.jit
def fval_xor(x, addr):
    cost = jnp.prod(x[addr], axis=1)
    return jnp.sum(cost)

@jax.jit
def fval(x, weight, eos, dft, idft, cnfs, xors):
    return weight * (fval_eo(x, eos, dft, idft) + 
                     fval_cnf(x, cnfs)) + fval_xor(x, xors)

@jax.jit
def verify_eo(x, addr):
    value = (x[addr] < 0)
    return (jnp.sum(value, axis = 1) != 1) + 0

@jax.jit
def verify_cnf(x, addr):
    value = -x[addr]                # all negative literals 
    return (jnp.min(value, axis = 1) > 0) + 0

@jax.jit
def verify_xor(x, xors):
    return (jnp.prod(x[xors], axis = 1) > 0) + 0

def run(filepath, tasks):
    t0 = time()
    CNF, EO, XOR, nv = read_hwcnf(filepath)
    weight = len(XOR) + 1

    eos, (dft, idft) = preprocess_eo(EO)
    cnfs = preprocess_cnf(CNF)
    xors = preprocess_xor(XOR)

    best_cost = len(XOR)

    pg = ProjectedGradient(fun=fval, projection=projection_box)
    def opt(x0, weight, eos, dft, idft, cnfs, xors):
        x0 = pg.run(x0, hyperparams_proj=(-1,1), 
                    weight=weight, eos=eos, 
                    dft=dft, idft=idft, 
                    cnfs=cnfs, xors=xors).params
        return x0, verify_eo(x0, eos), verify_cnf(x0, cnfs), verify_xor(x0, xors)
    p_opt = jax.jit(jax.pmap(opt, in_axes=(0, None, None, None, None, None, None)))

    tic = 0
    err = nv
    has_model = False
    while (time() - t0 < 300):
        key = jax.random.PRNGKey(tic)
        x0 = jax.random.truncated_normal(key, -1.0, 1.0, shape=(tasks, nv))

        # Optimization
        x0, cost_eo, cost_cnf, cost_xor = p_opt(x0, weight, eos, dft, idft, cnfs, xors)
        cost_eo  = cost_eo.sum(axis=1)
        cost_cnf = cost_cnf.sum(axis=1)
        cost_xor = cost_xor.sum(axis=1)
        cost = weight * (cost_eo + cost_cnf) + cost_xor
        if cost.min() < best_cost:
            best_cost = cost.min()
            print("{:.2f}/300.00 o {}".format(time()-t0, best_cost))
            sys.stdout.flush()
        if best_cost == 0:
            break
        tic += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type = str)
    args = parser.parse_args()

    if not args.filepath:
        print("File not found")
        exit(0)

    run(args.filepath, 32)

if __name__ == "__main__":
    main()