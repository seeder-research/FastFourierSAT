import os
import numpy as np
import random
import math
import copy
from optilog.encoders.pb import Encoder

def generate(n, lits):
    r = random.gauss(n/2, 1)
    l = math.ceil(r)
    if l > n/2 + 2:
        l = int(n/2 + 2)
    elif l < n/2 - 2:
        l = int(n/2 - 2)
    lits = random.sample(lits,l)
    lits.sort()
    return lits

def xor2cnf(xor):
    if len(xor) == 3:
        return [[-xor[0], -xor[1],  xor[2]], [ xor[0], -xor[1], -xor[2]],
               [-xor[0],  xor[1], -xor[2]], [ xor[0],  xor[1],  xor[2]]]
    if len(xor) == 2:
        return [[ xor[0],  xor[1]], [-xor[0], -xor[1]]]
    
def de_xor(llist, start):
    oldstart = start
    exactFlag = 0
    if len(llist) <= 3:
        return [list(llist),], start
    else:
        clist = []
        i = 0
        while True:
            if i >= len(llist):
                break
            elif i == len(llist)-1:
                exactFlag = 1
                break
            else:
                clist.append([-llist[i], llist[i + 1], start + 1])
                start+=1
                i+=2
    if exactFlag == 1:
        flist, _ = de_xor([llist[-1]] + list(range(oldstart + 1, start + 1)), start)
    else:
        flist, _ = de_xor(range(oldstart + 1, start + 1), start)
    return clist + flist, _
class parity(object):
    def __init__(self, nv, seed = 0):
        random.seed(seed)
        self.nv  = nv
        self.nv_ = 3 * nv
        self.mc  = 2 * nv            # Number of XOR constraints
        self.th  = int(nv / 2)       # Number of XOR constraints can be falsified
        
        self.lits = list(range(1, nv + 1))
        self.x = np.random.randint(0, 2, nv, dtype=bool)

        self.name  = "./hybrid/"+str(nv)+"/"+str(seed)+".hybrid"
        self.name_ = "./cnf/" +str(nv)+"/"+str(seed)+".cnf"

        self.DB      = []
        self.DB_cnf  = []
        self.DB_xor  = []
        self.DB_card = []
        self.generate_parity()

    def generate_parity(self):
        error = 0
        for i in range(self.mc):
            aux_var = int(self.nv + i + 1)
            self.DB_card.append(-aux_var)
            clause = generate(self.nv, self.lits)
            value = None
            for lit in clause:
                addr = abs(lit) - 1
                asgn = self.x[addr]
                if lit < 0:
                    asgn = not asgn
                if value == None:
                    value = asgn
                else:
                    value ^= asgn
            if value == False:
                clause[0] = -clause[0]
            self.DB.append(clause)
            clause = copy.deepcopy(clause)
            clause.append(aux_var)
            self.DB_xor.append(clause)
        # randomly choose self.th clauses to be falsified:
        err = random.sample(list(range(self.mc)), self.th-1)
        for i in err:
            self.DB[i][0]     = -self.DB[i][0]
            self.DB_xor[i][0] = -self.DB_xor[i][0] 
        self.convert()
        self.toHybrid()
        self.toCNF()

    def convert(self):
        self.nv_, cnfs = Encoder.at_least_k(self.DB_card, self.mc - self.th, max_var=self.nv_)
        for cnf in cnfs:
            self.DB_cnf.append(cnf)
        for clause in self.DB_xor:
            xors, self.nv_ = de_xor(clause, self.nv_ + 1)
            for xor in xors:
                cnfs = xor2cnf(xor)
                for cnf in cnfs:
                    self.DB_cnf.append(cnf)
    
    def toHybrid(self):
        with open(self.name, 'w') as f:
            f.write('c parity = %s \n' % ' '.join(map(str, self.x+0)))
            f.writelines('c e = 0.25 \n')
            f.writelines('p hybrid %d %d\n' % (self.nv, self.mc))
            f.writelines(['x %s 0\n' % ' '.join(map(str, c)) for c in self.DB])
            
    def toCNF(self):
        with open(self.name_, 'w') as f:
            f.write('c parity = %s \n' % ' '.join(map(str, self.x+0)))
            f.writelines('c e = 0.25 \n')
            f.writelines('p cnf %d %d\n' % (self.nv_, len(self.DB_cnf)))
            f.writelines(['%s 0\n' % ' '.join(map(str, c)) for c in self.DB_cnf])


for seed in range(100):
    for nv in [20, 30, 40, 50, 60]:
        F = parity(nv, seed)