def xor2cnf(xor):
    if len(xor)==3:
        return [[-xor[0],-xor[1],xor[2]], [xor[0],-xor[1],-xor[2]], [-xor[0],xor[1],-xor[2]],[xor[0],xor[1],xor[2]] ]
    if len(xor)==2:
        return [[xor[0],xor[1]],[-xor[0],-xor[1]]]
    
def de_xor(llist,start):
    oldstart = start
    exactFlag = 0
    if len(llist)<=3:
        return [list(llist),],start
    else:
        clist = []
        i = 0
        while True:
            if i>=len(llist):
                break
            elif i==len(llist)-1:
                exactFlag = 1
                break
            else:
                clist.append([-llist[i],llist[i+1],start+1])
                start+=1
                i+=2
    if exactFlag == 1:
        flist,_ = de_xor([llist[-1]] + list(range(oldstart+1,start+1)),start)
    else:
        flist,_ = de_xor(range(oldstart+1,start+1),start)
    return clist+flist,_