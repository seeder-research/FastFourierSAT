import numpy as np
from numpy import poly1d
from fractions import Fraction as F

def binom(n,k):
    if k==0: return 1
    ans = F(1)
    for i in range(n):
        ans = ans * (i+1)
        if i < k:
            ans = ans/(i+1)
        else:
            ans = ans/(n-i)
    return ans


def CardinalityFC(n,k):
    if n==1 and k==1:
        return [0,1]
    d = np.zeros(n + 1)
    if k==0:
        d[0]=-1
        return d
    templist = ([-1 for i in range(n-k)] + [1 for i in range(k-1)])
    templist = np.array(templist,dtype=object)
    
    temppoly = poly1d(templist,True) * (1-2*((k-1)%2))
    temparray = temppoly.c[::-1]
    temparray = [F(temparray[i],(2**(n-1))) for i in range(len(temparray))]
    binom_nk = binom(n-1,k-1)
    binom_ni_inv = [1 for i in range(n)]
    for i in range(0,n-1):
        binom_ni_inv[i+1] = binom_ni_inv[i] * (i+1) / (n-i-1)
    binocoef = [binom_nk * binom_ni_inv[i] for i in range(n)]
    temparray = [temparray[i] * binocoef[i] for i in range(len(temparray))]
    for j in range(n):
        d[j+1] = F(temparray[j])
    templist3 = []
    templist3.append(binom(n,k))
    for i in range(k,n):
        templist3.append(templist3[i-k] * (n-i) / (i+1) )
    d[0] = F(1 - F(int(sum(templist3)),(2 ** (n - 1))))
    return d