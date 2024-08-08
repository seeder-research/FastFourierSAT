from utils.gen_graph import *
from utils.gen_hash import *
from pysat.card import *

import argparse

def generate_hwcnf(grid, degree, num_hash, seed):
    homedir = "./"
    CNF, EO = generate_reg_graph(grid, degree, seed)
    nv = grid * degree
    topid = nv

    XOR = []
    var_list = gen_var_list(nv)

    random.seed(seed + num_hash)
    for i in range(num_hash):
        xor = gen_xor(var_list, int(nv//2))
        XOR.append(xor)
    
    num_of_clause = len(CNF) + len(XOR) + len(EO)
    hard_weight = len(XOR) + 1

    with open(homedir + "/hybrid/color_reg_n" + str(grid) + "_d" + str(degree) + "_x" + str(num_hash) + "_" + str(seed) + ".hybrid", 'w') as f:
        f.write('p hybrid %d %d %d\n' % (topid, num_of_clause, hard_weight))
        f.writelines(['h cnf %s 0\n' % ' '.join(map(str, c)) for c in CNF])
        f.writelines(['h eo %s 0\n' % ' '.join(map(str, c)) for c in EO])
        f.writelines(['1 x %s 0\n' % ' '.join(map(str, c)) for c in XOR])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, help='size')
    parser.add_argument('--degree', type=int, help='degree')
    parser.add_argument('--hash', type=int, help='hash')
    parser.add_argument('--num', type=int, help='num')
    args = parser.parse_args()
    if not args.size:
        size = 100
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
        generate_hwcnf(size, degree, hash, i)

if __name__ == "__main__":
    main()