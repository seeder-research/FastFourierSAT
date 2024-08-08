import random

def gen_var_list(n):
    var_list = []
    for i in range(n):
        var_list.append(i+1)
    return var_list

def gen_xor(vars, l):
    lits = random.sample(vars, l)
    lits.sort()
    return lits

def harden_xor(xor, topid):
    topid += 1
    xor.append(topid)
    return xor, topid