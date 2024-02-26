import jax
import jax.numpy as jnp
import numpy as np

from jaxopt import ProjectedGradient
from jaxopt.projection import projection_box

import argparse
from time import time
from utils.loader import Formula
from utils.card_coef import CardinalityFC

jax.config.update("jax_enable_x64", True)

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


def solve_cards(filepath, tasks, verbose):
    t0 = time()
    F = Formula()
    F.read_DIMACS(filepath)

    nv = F._n_var
    addr, sign, weight, (dft, idft) = preprocess_card(F._card_claus)

    pg = ProjectedGradient(fun=fval, projection=projection_box, maxiter=100, tol=1e-12)

    def opt(x0, addr, sign, weight, dft, idft):
        x0 = pg.run(x0, hyperparams_proj=(-1,1), addr=addr, sign=sign, weight=weight, dft=dft, idft=idft).params
        return x0, verify(x0, addr, sign)
    p_opt = jax.jit(jax.vmap(opt, in_axes=(0, None, None, None, None, None)))

    tic = 0
    err = nv

    while (err > 0) and ((time() - t0) < 60):
        if tic%3 == 0:
            key = jax.random.PRNGKey(tic)
            x0 = jax.random.uniform(key, (tasks, nv), minval = -1.0, maxval = 1.0)
        elif tic%3 == 2:
            x0 = -x0

        x0, res = p_opt(x0, addr, sign, weight, dft, idft)
        reward = res.sum(axis=0)
        weight = 0.6 * weight + 0.4 * reward/reward.max()
        err = res.sum(axis=1).min()
        tic += 1

    if verbose > 0:
        x = -jnp.sign(x0[jnp.argmin(res.sum(axis=1)),:]) * jnp.arange(1, nv+1)
        print(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type = str)
    parser.add_argument("--tasks", type = int, help = "Number of tasks that will parallelly run on a GPU")
    parser.add_argument("--verbose", type = int, help = "Print information")
    args = parser.parse_args()

    if not args.filepath:
        print("File not found")
        exit(0)

    if not args.tasks:
        tasks = 32
    else:
        tasks = args.tasks

    if not args.verbose:
        verbose = 1
    else:
        verbose = args.verbose


    solve_cards(args.filepath, tasks, verbose)

if __name__ == "__main__":
    main()