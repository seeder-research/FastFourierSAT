class Formula(object):
    def __init__(self):
        self._n_var = 0
        self._n_cls = 0
        self._hard  = []
        self._soft  = []
        self._weight = []
        

    def add_hard_clause(self, lits):
        self._hard.append(list(lits))

    def add_soft_clause(self, lits, weight):
        self._soft.append(list(lits))
        self._weight.append(weight)
    
    
    def read_DIMACS(self, filename, w=0):
        with open(filename, 'r') as f:
            for line in f:
                split = line.split()
                if len(line) == 0 or line[0] == 'c' or split[0] == '*':
                    continue
                if (len(split)>=2 and split[1] == '#variable=') or (split[0] == 'p'):    # Read p line
                    self._n_var = int(line.split()[2])
                elif (split[0] == 'h'):
                    lits = map(int, [line.split()[i] for i in range(1, len(split) - 1)])
                    lits = list(lits)
                    for l in lits:
                        if abs(l) > self._n_var:
                            self._n_var = abs(l)
                    self.add_hard_clause(lits)
                else:
                    weight = float(split[0])
                    lits = map(int, [line.split()[i] for i in range(1, len(split) - 1)])
                    lits = list(lits)
                    for l in lits:
                        if abs(l) > self._n_var:
                            self._n_var = abs(l)
                    self.add_soft_clause(lits, weight)