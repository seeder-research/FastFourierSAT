from pysat.card import *
from utils.loader import Formula
from utils.encodecnf import *
from utils.gen_hash import *

import argparse

def read_hybrid(path):
    F = Formula()
    F.read_DIMACS(path)
    return F._cnf_claus, F._eo_claus, F._xor_claus, F._n_var

def h2wcnf(grid, degree, num_hash, seed):
    homedir = "./"
    hardCNF, EO, softXOR, nv = read_hybrid(homedir + "/hybrid/color_reg_n" + str(grid) + "_d" + str(degree) + "_x" + str(num_hash) + "_" + str(seed) + ".hybrid")
    topid = nv

    # Encode Exact-1 constraints into hard clauses
    for eo in EO:
        cnfs = CardEnc.equals(lits=eo, encoding=0)
        for cnf in cnfs:
            hardCNF.append(cnf)

    # Encode soft XOR constraints into hard XOR constraints
    hardXOR = []
    softCNF = []
    for soft_xor in softXOR:
        hard_xor, topid = harden_xor(soft_xor, topid)
        softCNF.append([-topid])
        hardXOR.append(hard_xor)
    
    num_of_clause = len(hardCNF) + len(hardXOR) + len(softCNF)
    hard_weight = len(softCNF) + 1

    # Blow hard XOR constraints into hard CNF constraints
    for xor in hardXOR:
        blow_clauses, _ = de_xor(xor, topid)
        for b in blow_clauses:
            for l in b:
                if abs(l) > topid:
                    topid = abs(l)
            cnfs = xor2cnf(b)
            hardCNF += cnfs

    with open(homedir + "/wcnf/color_reg_n" + str(grid) + "_d" + str(degree) + "_x" + str(num_hash) + "_" + str(seed) + ".wcnf", 'w') as f:
        f.writelines(['h %s 0\n' % ' '.join(map(str, c)) for c in hardCNF])
        f.writelines(['1 %s 0\n' % ' '.join(map(str, c)) for c in softCNF])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, help='size')
    parser.add_argument('--degree', type=int, help='degree')
    parser.add_argument('--hash', type=int, help='hash')
    parser.add_argument('--num', type=int, help='num')
    args = parser.parse_args()
    if not args.size:
        size = 8
    else:
        size = args.size
    if not args.degree:
        degree = int(size//10)
    else:
        degree = args.degree
    if not args.hash:
        hash = size
    else:
        hash = args.hash
    if not args.num:
        num = 10
    else:
        num = args.num
    
    for i in range(num):
        h2wcnf(size, degree, hash, i)

if __name__ == "__main__":
    main()