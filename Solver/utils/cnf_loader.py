class Formula(object):
    def __init__(self):
        self._clauses  = []
    
    def read_DIMACS(self, filename, w=0):
        with open(filename, 'r') as f:
            for line in f:
                split = line.split()
                if len(line) == 0 or line[0] == 'c' or split[0] == 'p':
                    continue
                else:
                    lits = [int(val) for val in split]
                    self._clauses.append(lits)