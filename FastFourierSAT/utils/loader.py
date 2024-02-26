class Formula(object):
    def __init__(self):
        self._weighted  = 0
        self._n_var = 0
        self._n_cls = 0
        self._xor_claus  = []
        self._cnf_claus  = []
        self._xor_score  = []
        self._cnf_score  = []
        self._card_claus = []
        self._card_score = []
        self._card_k     = []
        

    def add_clause(self, lits, ctype):
        if self._weighted:
            weight = lits[0]
            lits = lits[1:]
        else:
            weight = 1
        if ctype == "d":
            self._card_claus.append(list(lits))
            self._card_score.append(weight)
        elif ctype == "x":
            self._xor_claus.append(list(lits))
            self._xor_score.append(weight)
        elif ctype == "c":
            self._cnf_claus.append(list(lits))
            self._cnf_score.append(weight)
    

    @property
    def card_clauses(self):
        return list(self._card_claus)


    @property
    def xor_clauses(self):
        return list(self._xor_claus)
    

    @property
    def cnf_clauses(self):
        return list(self._xor_claus)
    
    
    def read_DIMACS(self, filename, w=0):
        self._weighted = w
        with open(filename, 'r') as f:
            for line in f:
                split = line.split()
                if len(line) == 0 or line[0] == 'c' or line[0] == '*':
                    continue
                if (len(split)>=2 and split[1] == '#variable=') or (line[0] == 'p'):    # Read p line
                    self._n_var = int(line.split()[2])
                elif line[0] == 'd':                                                    # Cardinality
                    self._card_k.append(-int(line.split()[1]))
                    lits = map(int, [line.split()[i] for i in range(2, len(line.split()) - 1)])
                    lits = list(lits)
                    self.add_clause(lits, 'd')
                elif line[0] == 'x':
                    lits = map(int, [line.split()[i] for i in range(1, len(line.split()) - 1)])
                    lits = list(lits)
                    self.add_clause(lits, 'x')
                else:
                    lits = map(int, [line.split()[i] for i in range(0, len(line.split()) - 1)])
                    lits = list(lits)
                    self.add_clause(lits, 'c')