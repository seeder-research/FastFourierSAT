import os
import math
import random
from optilog.encoders.pb import Encoder

def generate_variable_list(n):
    var_list = []
    for i in range(n):
        var_list.append(i+1)
    return var_list

def pick_var(var_list, k):
    lits = random.sample(var_list, k)
    lits.sort(key=abs)
    return lits

def gen_card(var_list, k):
    lits = pick_var(var_list, k)
    k = math.ceil(k/2)
    if random.random() > 0.5:
        lits = [-x for x in lits]
    return ['d', k] + lits

class card(object):
    def __init__(self, nv, seed = 0):
        random.seed(seed)
        self.nv  = nv
        self.nv_ = nv
        self.lc  = int(0.2 * nv)       # Length of CARD constraints
        self.mc  = int(0.6 * nv)       # Number of CARD constraints
        self.var = generate_variable_list(nv)

        self.name  = "./hybrid/"+str(nv)+"/"+str(seed)+".hybrid"
        self.name_ = "./cnf/" +str(nv)+"/"+str(seed)+".cnf"

        self.cards  = []
        self.cards_ = []

    def generate(self):
        for i in range(self.mc):
            card = gen_card(self.var, self.lc)
            self.cards.append(card)
            self.nv_, cnfs = Encoder.at_least_k(card[2:], card[1], max_var=self.nv_)#, encoding='seqcounter')
            for cnf in cnfs:
                self.cards_.append(cnf)
        self.toCNF()
        self.toHybrid()

    def toHybrid(self):
        with open(self.name, 'w') as f:
            f.write('p hybrid %d %d\n' % (self.nv, len(self.cards)))
            f.writelines(['%s 0\n' % ' '.join(map(str, c)) for c in self.cards])

    def toCNF(self):
        with open(self.name_, 'w') as f:
            f.write('p cnf %d %d\n' % (self.nv_, len(self.cards_)))
            f.writelines('c cardinality constraints\n')
            f.writelines(['%s 0\n' % ' '.join(map(str, c)) for c in self.cards_])

for seed in range(100):
    for nv in [50, 100, 150, 200, 250]:
        gen = card(nv, seed)
        gen.generate()