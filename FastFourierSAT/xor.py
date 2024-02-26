import jax
import jax.numpy as jnp
import numpy as np
from jaxopt import ProjectedGradient
from jaxopt.projection import projection_box

import argparse
from time import time
from utils.loader import Formula

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
def fval(x, addr, sign, weight):
    x.at[0].set(1)
    value = sign * x[addr]
    cost = weight * jnp.prod(value, axis = 1)

    return jnp.sum(cost)

@jax.jit
def verify(x, addr, sign, weight):
    x.at[0].set(1)
    value = sign * x[addr]

    return weight * (jnp.prod(value, axis = 1) >= 0)

def solve_xor(filepath, tasks, verbose, tol, weighted):
    t0 = time()
    F = Formula()
    F.read_DIMACS(filepath, weighted)
    
    nv = F._n_var
    err = max(F._xor_score) * len(F._xor_claus)
    addr, sign = preprocess_xor(F._xor_claus)
    weight = jnp.array(F._xor_score)

    pg = ProjectedGradient(fun=fval, projection=projection_box, maxiter=100, tol=1e-12)

    def opt(x0, addr, sign, weight):
        x0 = pg.run(x0, hyperparams_proj=(-1,1), addr=addr, sign=sign, weight=weight).params
        return x0, verify(x0, addr, sign, weight)
    p_opt = jax.jit(jax.vmap(opt, in_axes=(0, None, None, None)))

    tic = 0
    best = max(F._xor_score) * len(F._xor_claus)

    while (err > tol) and ((time() - t0) < 60):
        if tic%2 == 0:
            key = jax.random.PRNGKey(tic)
            x0 = jax.random.uniform(key, (tasks, nv+1), minval = -1.0, maxval = 1.0)
        else:
            x0 = -x0

        x0, res = p_opt(x0, addr, sign, weight)
        err = res.sum(axis=1).min()
        tic += 1

        if best > err:
            best = err
            best_x = -jnp.sign(x0[jnp.argmin(res.sum(axis=1)),1:]) * jnp.arange(1, nv+1)

        if verbose > 0:
            print("o {}".format(int(err)))
        if verbose > 1:
            print("c Iter: {} Time: {:.2f} s".format(tic, time()-t0))

    if (err > tol):
        print("Reach time limit, the lowest error is {}".format(int(best)))

    print(best_x)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type = str)
    parser.add_argument("--tasks", type = int, help = "Number of tasks that will parallelly run on a GPU")
    parser.add_argument("--verbose", type = int, help = "Print information")
    parser.add_argument("--tolerance", type = int, help = "Number of constraints allowed to be violated")
    parser.add_argument("--weighted", type = int, help = "Whether the formula is weighted")
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

    if not args.tolerance:
        tol = 0
    else:
        tol = args.tolerance

    if not args.weighted:
        w = 0
    else:
        w = 1

    solve_xor(args.filepath, tasks, verbose, tol, w)

if __name__ == "__main__":
    main()