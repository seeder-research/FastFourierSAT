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
from ..utils.coef import CardinalityFC

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", "tmp/jax-cache")

def fft(n, k):
    coef = jnp.array(CardinalityFC(n, k))
    omega = jnp.exp(2j * jnp.pi * jnp.arange(n+1) / (n+1)).reshape(n+1, 1)
    inv_dft = omega ** jnp.arange(n+1) / (n+1)
    omega = jnp.exp(-2j * jnp.pi * jnp.arange(len(inv_dft)) / len(inv_dft)).reshape(n+1, 1, 1)
    return jnp.broadcast_to(omega, (n+1,1,1)), coef[::-1]@inv_dft

def preprocess_card(card_clauses):
    card = np.array(card_clauses)
    m, n = card.shape
    addr = np.abs(card) - 1
    sign = np.sign(card)
    weight = np.ones(m)
    return jnp.array(addr, dtype = int), jnp.array(sign, dtype=int), jnp.array(weight, dtype=float), fft(n, int(n/2))

@jax.jit
def fval(x, addr, sign, weight, dft, idft):
    value = sign * x[addr]
    prod = jnp.prod(dft+value, axis=2)
    cost = weight * (idft@prod).real
    return jnp.sum(cost)

@jax.jit
def verify(x, addr, sign):
    value = jnp.sign(sign * x[addr])
    return jnp.sum(value, axis = 1) > 0

def solve_cards(filepath, tasks):
    F = Formula()
    F.read_DIMACS(filepath)
    nv = F._n_var
    addr, sign, weight, (dft, idft) = preprocess_card(F._card_claus)

    pg = ProjectedGradient(fun=fval, projection=projection_box, maxiter=100)
    def opt(x0, addr, sign, weight, dft, idft):
        x0 = pg.run(x0, hyperparams_proj=(-1,1), addr=addr, sign=sign, weight=weight, dft=dft, idft=idft).params
        return x0, verify(x0, addr, sign)
    p_opt = jax.jit(jax.vmap(opt, in_axes=(0, None, None, None, None, None)))

    tic = 0
    err = nv
    while (err > 0):
        key = jax.random.PRNGKey(tic)
        x0 = jax.random.uniform(key, (tasks, nv), minval = -1.0, maxval = 1.0)
        x0, res = p_opt(x0, addr, sign, weight, dft, idft)
        reward = res.sum(axis=0)
        weight = 0.9 * weight + 0.1 * reward/reward.max()
        err = res.sum(axis=1).min()
        tic += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type = str)
    args = parser.parse_args()

    if not args.filepath:
        print("File not found")
        exit(0)

    solve_cards(args.filepath, 64)

if __name__ == "__main__":
    main()